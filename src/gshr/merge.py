from __future__ import annotations

import numpy as np

from .graph_stats import compute_boundary_and_internal_edges, neighbor_label_agreement
from .hyperbox import HyperboxRecord


def enclosing_bounds(lower_a: np.ndarray, upper_a: np.ndarray, lower_b: np.ndarray, upper_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    new_lower = np.minimum(lower_a, lower_b)
    new_upper = np.maximum(upper_a, upper_b)
    return new_lower, new_upper


def volume_inflation_ratio(box_a: HyperboxRecord, box_b: HyperboxRecord) -> float:
    new_lower, new_upper = enclosing_bounds(box_a.lower, box_a.upper, box_b.lower, box_b.upper)
    merged_volume = float(np.prod(np.maximum(new_upper - new_lower, 0.0)))
    current_volume = float(box_a.volume() * box_b.volume())
    if current_volume <= 0.0:
        return 0.0
    return merged_volume / current_volume


def graph_cohesion_score(
    box_a: HyperboxRecord,
    box_b: HyperboxRecord,
    edge_index: np.ndarray | None = None,
    node_labels: dict[int, int] | None = None,
) -> float:
    if edge_index is None or edge_index.size == 0:
        return 1.0
    members = set(box_a.members) | set(box_b.members)
    if not members:
        return 1.0
    internal, boundary = compute_boundary_and_internal_edges(members, edge_index)
    total = internal + boundary
    if total == 0:
        return 1.0
    cohesion = internal / total
    if node_labels is None:
        return float(cohesion)
    agreement = neighbor_label_agreement(members, node_labels, edge_index)
    return float(0.5 * cohesion + 0.5 * agreement)


def validation_purity(
    box: HyperboxRecord,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
    label_tolerance: float = 0.0,
) -> float:
    if validation_x is None or validation_y is None:
        return 1.0
    if validation_x.size == 0 or validation_y.size == 0:
        return 1.0
    contained = (validation_x >= box.lower).all(axis=1) & (validation_x <= box.upper).all(axis=1)
    if not np.any(contained):
        return 1.0
    labels = validation_y[contained]
    if labels.size == 0:
        return 1.0
    counts = np.bincount(labels.astype(int), minlength=int(labels.max()) + 1)
    return float(counts[box.label] / labels.size) if box.label < counts.size else 0.0


def validation_surrogate_loss(
    box: HyperboxRecord,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
) -> float:
    purity = validation_purity(box, validation_x, validation_y)
    return float(1.0 - purity)


def can_merge_hyperboxes(
    box_a: HyperboxRecord,
    box_b: HyperboxRecord,
    max_side_length: float = 1.0,
    volume_limit: float = 1e6,
    edge_index: np.ndarray | None = None,
    node_labels: dict[int, int] | None = None,
    min_graph_cohesion: float = 0.0,
    validation_x: np.ndarray | None = None,
    validation_y: np.ndarray | None = None,
    min_validation_purity: float = 0.0,
) -> bool:
    if box_a.label != box_b.label:
        return False
    new_lower, new_upper = enclosing_bounds(box_a.lower, box_a.upper, box_b.lower, box_b.upper)
    if np.any(new_upper - new_lower > max_side_length):
        return False
    if volume_inflation_ratio(box_a, box_b) > volume_limit:
        return False
    if np.any(np.isnan(new_upper)) or np.any(np.isnan(new_lower)):
        return False
    if edge_index is not None and edge_index.size > 0:
        score = graph_cohesion_score(box_a, box_b, edge_index=edge_index, node_labels=node_labels)
        if score < min_graph_cohesion:
            return False
    if validation_x is not None and validation_y is not None:
        merged = HyperboxRecord(
            box_id=-1,
            lower=new_lower,
            upper=new_upper,
            label=box_a.label,
            members=set(box_a.members) | set(box_b.members),
        )
        purity = validation_purity(merged, validation_x, validation_y)
        if purity < min_validation_purity:
            return False
    return True


def merge_boxes(box_a: HyperboxRecord, box_b: HyperboxRecord, box_id: int = 0) -> HyperboxRecord:
    if not can_merge_hyperboxes(box_a, box_b):
        raise ValueError('The two hyperboxes cannot be merged under the configured constraints.')
    new_lower, new_upper = enclosing_bounds(box_a.lower, box_a.upper, box_b.lower, box_b.upper)
    merged = HyperboxRecord(
        box_id=box_id,
        lower=new_lower,
        upper=new_upper,
        label=box_a.label,
        members=set(box_a.members) | set(box_b.members),
        status='merged',
        metadata={'parent_ids': [box_a.box_id, box_b.box_id]},
    )
    return merged
