from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .graph_stats import conductance
from .hyperbox import HyperboxRecord
from .merge import can_merge_hyperboxes, graph_cohesion_score, merge_boxes, validation_purity, volume_inflation_ratio
from .prune import prune_boxes


class GraphStructureAwareRefinement:
    def __init__(self, boxes: Iterable[HyperboxRecord], edge_index: np.ndarray | None = None, node_labels: dict[int, int] | None = None) -> None:
        self.boxes = list(boxes)
        self.edge_index = edge_index
        self.node_labels = node_labels or {}

    def _merge_score(self, box_a: HyperboxRecord, box_b: HyperboxRecord) -> float:
        geometry = float(1.0 / (1.0 + max(volume_inflation_ratio(box_a, box_b), 1e-12)))
        if self.edge_index is not None and self.edge_index.size > 0:
            members = set(box_a.members) | set(box_b.members)
            if members:
                geometry += 0.5 * graph_cohesion_score(box_a, box_b, edge_index=self.edge_index, node_labels=self.node_labels)
                geometry -= 0.25 * conductance(members, self.edge_index)
        return geometry

    def _validation_objective(
        self,
        box_a: HyperboxRecord,
        box_b: HyperboxRecord,
        validation_embeddings: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> float:
        if validation_embeddings is None or validation_labels is None:
            return 0.0
        merged = merge_boxes(box_a, box_b, box_id=-1)
        merged_purity = validation_purity(merged, validation_embeddings, validation_labels)
        left_purity = validation_purity(box_a, validation_embeddings, validation_labels)
        right_purity = validation_purity(box_b, validation_embeddings, validation_labels)
        if merged_purity <= 0.0:
            return -1.0
        gain = merged_purity - max(left_purity, right_purity)
        volume_penalty = min(volume_inflation_ratio(box_a, box_b), 1.0)
        return float(0.75 * gain + 0.25 * (1.0 - volume_penalty))

    def merge(
        self,
        max_side_length: float = 1.0,
        volume_limit: float = 1e6,
        min_graph_cohesion: float = 0.0,
        validation_embeddings: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        min_validation_purity: float = 0.0,
        validation_objective: Callable[[list[HyperboxRecord]], float] | None = None,
        min_objective_gain: float = 0.0,
    ) -> list[HyperboxRecord]:
        if not self.boxes:
            return []
        remaining = list(self.boxes)
        merged: list[HyperboxRecord] = []
        while remaining:
            current = remaining.pop(0)
            best_idx = None
            best_match = None
            best_score = float('-inf')
            for idx, other in enumerate(remaining):
                if current.label != other.label:
                    continue
                if not can_merge_hyperboxes(
                    current,
                    other,
                    max_side_length=max_side_length,
                    volume_limit=volume_limit,
                    edge_index=self.edge_index,
                    node_labels=self.node_labels,
                    min_graph_cohesion=min_graph_cohesion,
                    validation_x=validation_embeddings,
                    validation_y=validation_labels,
                    min_validation_purity=min_validation_purity,
                ):
                    continue
                score = self._merge_score(current, other)
                score += self._validation_objective(current, other, validation_embeddings, validation_labels)
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    best_match = other
            if best_match is None:
                merged.append(current)
                continue
            merged_box = merge_boxes(current, best_match, box_id=max((box.box_id for box in merged), default=-1) + 1)
            if validation_objective is not None:
                current_score = validation_objective([current, best_match])
                merged_score = validation_objective([merged_box])
                if merged_score < current_score + min_objective_gain:
                    merged.append(current)
                    continue
            merged.append(merged_box)
            del remaining[best_idx]
        return merged

    def prune(
        self,
        rule: dict[str, float],
        validation_embeddings: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        validation_objective: Callable[[list[HyperboxRecord]], float] | None = None,
        min_objective_gain: float = 0.0,
    ) -> list[HyperboxRecord]:
        return prune_boxes(
            self.boxes,
            rule,
            edge_index=self.edge_index,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            validation_objective=validation_objective,
            min_objective_gain=min_objective_gain,
        )
