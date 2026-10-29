from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from hbbrain.numerical_data.incremental_learner.fmnn import FMNNClassifier


@dataclass
class Hyperbox:
    lower: np.ndarray
    upper: np.ndarray
    label: int
    members: list[int]

    @property
    def volume(self) -> float:
        span = np.maximum(self.upper - self.lower, 0.0)
        return float(np.prod(span)) if span.size > 0 else 0.0

    def contains(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(np.all(x >= self.lower) and np.all(x <= self.upper))


class SimpleFMNN:
    """Compatibility wrapper around hbbrain's FMNN implementation."""

    def __init__(self, theta: float = 0.3, gamma: float = 1.0) -> None:
        self.theta = float(theta)
        self.gamma = float(gamma)
        self.model: FMNNClassifier | None = None
        self.boxes: list[Hyperbox] = []
        self.classes_: np.ndarray | None = None

    def _sync_from_model(self) -> None:
        if self.model is None:
            self.boxes = []
            self.classes_ = np.asarray([], dtype=int)
            return
        v = np.asarray(self.model.V, dtype=float)
        w = np.asarray(self.model.W, dtype=float)
        c = np.asarray(self.model.C, dtype=int)
        self.boxes = [
            Hyperbox(lower=v[i].copy(), upper=w[i].copy(), label=int(c[i]), members=[int(i)])
            for i in range(len(c))
        ]
        self.classes_ = np.asarray(np.unique(c), dtype=int)

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SimpleFMNN':
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError('Feature matrix and labels must have the same length.')
        self.model = FMNNClassifier(theta=self.theta, gamma=self.gamma)
        self.model.fit(x, y)
        self._sync_from_model()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError('Model has not been fitted.')
        return np.asarray(self.model.predict(x), dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError('Model has not been fitted.')
        proba = self.model.predict_proba(x)
        return np.asarray(proba, dtype=float)

    def get_boxes(self) -> list[Hyperbox]:
        if self.model is None:
            return []
        self._sync_from_model()
        return list(self.boxes)

    def get_hyperbox_count_by_class(self) -> dict[int, int]:
        if self.model is None:
            return {}
        counts: dict[int, int] = {}
        for box in self.get_boxes():
            counts[box.label] = counts.get(box.label, 0) + 1
        return counts

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError('Model has not been fitted.')
        payload = [
            {'label': int(label), 'lower': row[0].tolist(), 'upper': row[1].tolist()}
            for row, label in zip(np.stack([self.model.V, self.model.W], axis=1), self.model.C)
        ]
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh)

    def load(self, path: str) -> 'SimpleFMNN':
        with open(path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        if not payload:
            self.model = FMNNClassifier(theta=self.theta, gamma=self.gamma)
            self.boxes = []
            self.classes_ = np.asarray([], dtype=int)
            return self
        V = np.asarray([item['lower'] for item in payload], dtype=float)
        W = np.asarray([item['upper'] for item in payload], dtype=float)
        C = np.asarray([int(item['label']) for item in payload], dtype=int)
        self.model = FMNNClassifier(theta=self.theta, gamma=self.gamma, V=V, W=W, C=C)
        self._sync_from_model()
        return self


def create_fmnn_adapter(theta: float = 0.3, gamma: float = 1.0) -> SimpleFMNN:
    return SimpleFMNN(theta=theta, gamma=gamma)
