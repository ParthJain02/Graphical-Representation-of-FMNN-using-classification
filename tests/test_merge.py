import numpy as np

from src.gshr.hyperbox import HyperboxRecord
from src.gshr.merge import can_merge_hyperboxes, merge_boxes
from src.gshr.refinement import GraphStructureAwareRefinement


def test_geometry_merge_accepts_compatible_boxes():
    a = HyperboxRecord(box_id=1, lower=np.array([0.0, 0.0]), upper=np.array([0.6, 0.6]), label=0, members={1, 2})
    b = HyperboxRecord(box_id=2, lower=np.array([0.5, 0.5]), upper=np.array([1.0, 1.0]), label=0, members={3, 4})
    assert can_merge_hyperboxes(a, b, max_side_length=1.0)
    merged = merge_boxes(a, b, box_id=10)
    assert merged.label == 0
    assert merged.members == {1, 2, 3, 4}


def test_geometry_merge_rejects_incompatible_class_labels():
    a = HyperboxRecord(box_id=1, lower=np.array([0.0]), upper=np.array([1.0]), label=0, members={1})
    b = HyperboxRecord(box_id=2, lower=np.array([0.2]), upper=np.array([2.0]), label=1, members={2})
    assert not can_merge_hyperboxes(a, b, max_side_length=1.0)


def test_graph_aware_refinement_merges_connected_hyperboxes():
    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=int)
    a = HyperboxRecord(box_id=1, lower=np.array([0.0]), upper=np.array([0.5]), label=0, members={0, 1})
    b = HyperboxRecord(box_id=2, lower=np.array([0.6]), upper=np.array([1.0]), label=0, members={2})
    refinement = GraphStructureAwareRefinement([a, b], edge_index=edge_index, node_labels={0: 0, 1: 0, 2: 0})
    merges = refinement.merge(max_side_length=1.5)
    assert len(merges) == 1
    assert merges[0].label == 0
    assert merges[0].members == {0, 1, 2}
