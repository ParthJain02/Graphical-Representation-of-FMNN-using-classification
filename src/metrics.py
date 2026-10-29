from __future__ import annotations

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('True and predicted labels must have the same shape.')
    return float(np.mean(y_true == y_pred))


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        denom = 2 * tp + fp + fn
        if denom == 0:
            scores.append(0.0)
        else:
            scores.append((2 * tp) / denom)
    return float(np.mean(scores))
