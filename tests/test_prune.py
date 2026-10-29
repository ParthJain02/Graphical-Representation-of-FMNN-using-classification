import numpy as np

from src.gshr.hyperbox import HyperboxRecord
from src.gshr.prune import prune_boxes


def test_pruning_never_removes_last_box_for_a_class():
    boxes = [
        HyperboxRecord(box_id=1, lower=np.array([0.0]), upper=np.array([1.0]), label=0, members={1}),
        HyperboxRecord(box_id=2, lower=np.array([4.0]), upper=np.array([5.0]), label=1, members={2}),
    ]
    kept = prune_boxes(boxes, {'max_loss_increase': 0.1})
    assert len(kept) == len(boxes)
