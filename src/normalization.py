from __future__ import annotations

import numpy as np


class MinMaxNormalizer:
    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray) -> 'MinMaxNormalizer':
        x = np.asarray(x, dtype=float)
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        return self

    def transform(self, x: np.ndarray, clip: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.min_ is None or self.max_ is None:
            raise ValueError('Normalizer has not been fitted.')
        out = np.zeros_like(x, dtype=float)
        for j in range(x.shape[1]):
            denom = max(float(self.max_[j] - self.min_[j]), self.epsilon)
            out[:, j] = (x[:, j] - self.min_[j]) / denom
            if clip:
                out[:, j] = np.clip(out[:, j], 0.0, 1.0)
        return out

    def fit_transform(self, x: np.ndarray, clip: bool = False) -> np.ndarray:
        return self.fit(x).transform(x, clip=clip)
