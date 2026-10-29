import numpy as np

from src.fmnn_adapter import SimpleFMNN


def test_fmnn_adapter_fits_and_predicts():
    x = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model = SimpleFMNN(theta=0.3, gamma=1.0)
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape == y.shape
    assert len(model.get_boxes()) == 2
    counts = model.get_hyperbox_count_by_class()
    assert counts[0] >= 1 and counts[1] >= 1
