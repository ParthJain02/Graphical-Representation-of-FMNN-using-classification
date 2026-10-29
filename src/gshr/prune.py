from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .graph_stats import compute_boundary_and_internal_edges
from .hyperbox import HyperboxRecord


def safe_prune_candidate(
    box: HyperboxRecord,
    rule: dict[str, float],
    class_coverage: dict[int, int],
    validation_loss: float = 0.0,
    edge_index: np.ndarray | None = None,
    validation_embeddings: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
) -> bool:
    coverage = class_coverage.get(box.label, 0)
    min_class_coverage = int(rule.get('min_class_coverage', 1))
    if coverage <= min_class_coverage:
        return False
    if validation_loss > rule.get('max_loss_increase', 0.1):
        return False
    if edge_index is not None and edge_index.size > 0 and box.members:
        internal, boundary = compute_boundary_and_internal_edges(set(box.members), edge_index)
        total = internal + boundary
        if total > 0:
            cohesion = internal / total
            if cohesion < rule.get('min_graph_cohesion', 0.0):
                return False
    if validation_embeddings is not None and validation_labels is not None:
        contained = (validation_embeddings >= box.lower).all(axis=1) & (validation_embeddings <= box.upper).all(axis=1)
        if np.any(contained):
            labels = validation_labels[contained]
            if labels.size > 0:
                same = np.sum(labels == box.label)
                purity = same / labels.size
                if purity < rule.get('min_prune_purity', 0.0):
                    return False
                if validation_loss > rule.get('max_validation_loss', 0.5):
                    return False
    return True


def prune_boxes(
    boxes: Iterable[HyperboxRecord],
    rule: dict[str, float],
    edge_index: np.ndarray | None = None,
    validation_embeddings: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    validation_objective: Callable[[list[HyperboxRecord]], float] | None = None,
    min_objective_gain: float = 0.0,
) -> list[HyperboxRecord]:
    current = list(boxes)
    if not current:
        return []
    while True:
        class_coverage: dict[int, int] = {}
        for box in current:
            class_coverage[box.label] = class_coverage.get(box.label, 0) + 1
        candidates: list[tuple[float, HyperboxRecord, list[HyperboxRecord]]] = []
        current_score = validation_objective(current) if validation_objective is not None else 0.0
        for box in current:
            validation_loss = 0.0
            if validation_embeddings is not None and validation_labels is not None:
                contained = (validation_embeddings >= box.lower).all(axis=1) & (validation_embeddings <= box.upper).all(axis=1)
                if np.any(contained):
                    labels = validation_labels[contained]
                    if labels.size > 0:
                        same = np.sum(labels == box.label)
                        validation_loss = 1.0 - (same / labels.size)
            can_prune = safe_prune_candidate(
                box,
                rule,
                class_coverage,
                validation_loss=validation_loss,
                edge_index=edge_index,
                validation_embeddings=validation_embeddings,
                validation_labels=validation_labels,
            )
            if not can_prune:
                continue
            candidate = [item for item in current if item is not box]
            if validation_objective is None:
                candidates.append((0.0, box, candidate))
                continue
            pruned_score = validation_objective(candidate)
            gain = pruned_score - current_score
            if gain > min_objective_gain:
                candidates.append((gain, box, candidate))
        if not candidates:
            return current
        _, _, current = max(candidates, key=lambda item: item[0])
