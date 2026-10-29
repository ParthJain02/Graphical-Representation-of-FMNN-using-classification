from __future__ import annotations

import numpy as np


def extract_embeddings(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, 'embed'):
        return np.asarray(model.embed(x), dtype=float)
    if hasattr(model, 'forward'):
        return np.asarray(model.forward(x), dtype=float)
    raise TypeError('Model does not expose embeddings.')
