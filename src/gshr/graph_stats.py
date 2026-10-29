from __future__ import annotations

from typing import Iterable

import numpy as np


def node_membership_map(members: Iterable[int]) -> set[int]:
    return set(members)


def compute_boundary_and_internal_edges(members: set[int], edge_index: np.ndarray) -> tuple[int, int]:
    members = set(members)
    if edge_index.size == 0:
        return 0, 0
    internal = 0
    boundary = 0
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u in members and v in members:
            internal += 1
        elif u in members or v in members:
            boundary += 1
    return internal, boundary


def conductance(members: set[int], edge_index: np.ndarray) -> float:
    if not members:
        return 0.0
    internal, boundary = compute_boundary_and_internal_edges(members, edge_index)
    denom = 2 * internal + boundary + 1e-12
    return float(boundary / denom)


def neighbor_label_agreement(members: set[int], labels: dict[int, int], edge_index: np.ndarray) -> float:
    if not members:
        return 0.0
    same = 0
    total = 0
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u in members and v in members:
            total += 1
            same += 1 if labels.get(u, -1) == labels.get(v, -1) else 0
    return float(same / max(total, 1))


def degree_volume(members: set[int], degree: dict[int, float]) -> float:
    return float(sum(degree.get(node, 0.0) for node in members))
