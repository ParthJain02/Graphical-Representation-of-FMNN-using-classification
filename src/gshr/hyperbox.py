from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HyperboxRecord:
    box_id: int
    lower: np.ndarray
    upper: np.ndarray
    label: int
    members: set[int] = field(default_factory=set)
    status: str = 'active'
    metadata: dict[str, Any] = field(default_factory=dict)

    def volume(self) -> float:
        span = np.maximum(self.upper - self.lower, 0.0)
        return float(np.prod(span)) if span.size > 0 else 0.0

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))

    def class_purity(self, labels: dict[int, int]) -> float:
        if not self.members:
            return 0.0
        counts: dict[int, int] = {}
        for node_id in self.members:
            label = labels.get(node_id, self.label)
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return max(counts.values()) / total
