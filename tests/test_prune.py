"""Unit tests for graph-structure-aware hyperbox pruning."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import scipy.sparse as sp

from src.fmnn import FMNN, HyperBox
from src.merge import GraphContext
from src.prune import _compute_importance_score, _fast_confusion_metrics, geometry_only_prune, gshr_prune


@pytest.fixture
def pruning_setup() -> dict:
    """Fixture with a 2-class, 4-box FMNN and validation data."""
    fmnn = FMNN(theta=0.5, gamma=4.0)
    fmnn.n_dims = 2
    fmnn.num_classes = 2

    # Class 0: Box 1 (main), Box 2 (redundant clone)
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.1, 0.1]), w=np.array([0.3, 0.3]), cls=0, members=[0, 1])
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.12, 0.12]), w=np.array([0.28, 0.28]), cls=0, members=[2])

    # Class 1: Box 3 (only box)
    fmnn.boxes[3] = HyperBox(id=3, v=np.array([0.7, 0.7]), w=np.array([0.9, 0.9]), cls=1, members=[3])
    fmnn.next_id = 4

    Z_train = np.array([
        [0.15, 0.15],
        [0.25, 0.25],
        [0.20, 0.20],
        [0.80, 0.80],
    ])
    y_train = np.array([0, 0, 0, 1])
    train_ids = [0, 1, 2, 3]

    Z_val = np.array([
        [0.18, 0.18],
        [0.22, 0.22],
        [0.82, 0.82],
    ])
    y_val = np.array([0, 0, 1])

    # Adjacency with triangle on class 0 and disconnected class 1
    edges = [(0, 1), (1, 2), (2, 0)]
    row = [e[0] for e in edges] + [e[1] for e in edges]
    col = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(row), dtype=np.float32)
    adj_csr = sp.coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
    degree = np.array(adj_csr.sum(axis=1), dtype=np.int64).flatten()
    train_mask = np.ones(4, dtype=bool)

    ctx = GraphContext(adj_csr=adj_csr, degree=degree, y=y_train, train_mask=train_mask)
    cfg_prune = {
        "mu1": 0.25,
        "mu2": 0.25,
        "mu3": 0.20,
        "mu4": 0.20,
        "mu5": 0.10,
        "tau_prune": 0.05,
        "min_boxes_per_class": 1,
    }

    return {
        "fmnn": fmnn,
        "ctx": ctx,
        "cfg": cfg_prune,
        "Z_train": Z_train,
        "y_train": y_train,
        "train_ids": train_ids,
        "Z_val": Z_val,
        "y_val": y_val,
    }


def test_single_box_class_never_removed(pruning_setup: dict) -> None:
    """A class with min_boxes_per_class boxes should never lose its last box."""
    fmnn = FMNN(theta=0.5, gamma=4.0)
    fmnn.n_dims = 2
    fmnn.num_classes = 2
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.1, 0.1]), w=np.array([0.3, 0.3]), cls=0, members=[0])
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.7, 0.7]), w=np.array([0.9, 0.9]), cls=1, members=[1])

    p = pruning_setup
    pruned_fmnn, log = gshr_prune(
        fmnn, p["ctx"], p["cfg"],
        p["Z_train"][:2], p["y_train"][:2], [0, 1],
        p["Z_val"], p["y_val"],
    )

    assert pruned_fmnn.num_boxes() == 2
    assert log.pruned_count == 0


def test_redundant_box_removed(pruning_setup: dict) -> None:
    """A redundant box (Box 2) is removed while validation metrics remain intact."""
    p = pruning_setup
    pruned_fmnn, log = gshr_prune(
        p["fmnn"], p["ctx"], p["cfg"],
        p["Z_train"], p["y_train"], p["train_ids"],
        p["Z_val"], p["y_val"],
    )

    assert log.boxes_before == 3
    assert log.boxes_after == 2
    assert log.pruned_count == 1
    assert 2 not in pruned_fmnn.boxes
    assert 1 in pruned_fmnn.boxes
    assert 3 in pruned_fmnn.boxes
    assert np.isclose(log.val_macro_f1_before, log.val_macro_f1_after)


def test_critical_support_box_not_removed() -> None:
    """A box that provides sole support for validation points must NOT be removed."""
    fmnn = FMNN(theta=0.3, gamma=4.0)
    fmnn.n_dims = 2
    fmnn.num_classes = 2

    # Class 0: Box 1 at (0.1, 0.1), Box 2 at (0.8, 0.8)
    fmnn.boxes[1] = HyperBox(id=1, v=np.array([0.05, 0.05]), w=np.array([0.15, 0.15]), cls=0, members=[0])
    fmnn.boxes[2] = HyperBox(id=2, v=np.array([0.75, 0.75]), w=np.array([0.85, 0.85]), cls=0, members=[1])
    # Class 1: Box 3 near Box 1, Box 4 near Box 2
    fmnn.boxes[3] = HyperBox(id=3, v=np.array([0.18, 0.18]), w=np.array([0.25, 0.25]), cls=1, members=[2])
    fmnn.boxes[4] = HyperBox(id=4, v=np.array([0.65, 0.65]), w=np.array([0.72, 0.72]), cls=1, members=[3])

    Z_train = np.array([[0.1, 0.1], [0.8, 0.8], [0.22, 0.22], [0.68, 0.68]])
    y_train = np.array([0, 0, 1, 1])
    train_ids = [0, 1, 2, 3]

    # Val points: each box uniquely supports one validation point
    Z_val = np.array([[0.1, 0.1], [0.8, 0.8], [0.22, 0.22], [0.68, 0.68]])
    y_val = np.array([0, 0, 1, 1])

    adj = sp.csr_matrix((4, 4), dtype=np.float32)
    ctx = GraphContext(adj_csr=adj, degree=np.zeros(4, dtype=np.int64), y=y_train, train_mask=np.ones(4, dtype=bool))
    cfg = {"tau_prune": 0.05, "min_boxes_per_class": 1}

    pruned_fmnn, log = gshr_prune(
        fmnn, ctx, cfg, Z_train, y_train, train_ids, Z_val, y_val,
    )

    # Each box provides critical support; removing any box flips its validation point to competing class,
    # dropping recall and macro-F1 beyond tolerance. Hence all boxes must be kept!
    assert pruned_fmnn.num_boxes() == 4
    assert log.pruned_count == 0


def test_prune_determinism(pruning_setup: dict) -> None:
    """Running gshr_prune twice gives identical box sets and logs."""
    p = pruning_setup
    fmnn1, log1 = gshr_prune(
        p["fmnn"], p["ctx"], p["cfg"],
        p["Z_train"], p["y_train"], p["train_ids"],
        p["Z_val"], p["y_val"],
    )
    fmnn2, log2 = gshr_prune(
        p["fmnn"], p["ctx"], p["cfg"],
        p["Z_train"], p["y_train"], p["train_ids"],
        p["Z_val"], p["y_val"],
    )

    assert fmnn1.num_boxes() == fmnn2.num_boxes()
    assert sorted(fmnn1.boxes.keys()) == sorted(fmnn2.boxes.keys())
    assert log1.pruned_count == log2.pruned_count


def test_fast_score_recomputation_equals_from_scratch(pruning_setup: dict) -> None:
    """Fast score recomputation matches fmnn.class_scores on any reduced box set."""
    p = pruning_setup
    fmnn: FMNN = p["fmnn"]
    Z_val = p["Z_val"]

    # Precompute validation matrix
    box_list = sorted(fmnn.boxes.values(), key=lambda b: b.id)
    val_mem = fmnn.membership_matrix(Z_val, box_list)
    box_classes = np.array([b.cls for b in box_list])
    id_to_col = {b.id: idx for idx, b in enumerate(box_list)}

    active_subsets = [
        {1, 2, 3},
        {1, 3},
        {2, 3},
        {3},
    ]

    for subset in active_subsets:
        # Fast
        cols = [id_to_col[bid] for bid in sorted(subset)]
        act_mem = val_mem[:, cols]
        act_cls = box_classes[cols]
        fast_scores = np.zeros((len(Z_val), fmnn.num_classes))
        for c in range(fmnn.num_classes):
            c_mask = (act_cls == c)
            if np.any(c_mask):
                fast_scores[:, c] = np.max(act_mem[:, c_mask], axis=1)

        # From scratch
        reduced_boxes = [fmnn.boxes[bid] for bid in sorted(subset)]
        scratch_scores = fmnn.class_scores(Z_val, boxes=reduced_boxes)

        np.testing.assert_allclose(fast_scores, scratch_scores, atol=1e-12)


def test_importance_terms_bounded_in_01(pruning_setup: dict) -> None:
    """All importance score terms must lie within [0, 1]."""
    p = pruning_setup
    fmnn: FMNN = p["fmnn"]
    ctx = p["ctx"]
    Z_train = p["Z_train"]
    node_to_train_idx = {nid: idx for idx, nid in enumerate(p["train_ids"])}

    for bid, box in fmnn.boxes.items():
        imp, terms = _compute_importance_score(
            box=box,
            fmnn=fmnn,
            ctx=ctx,
            Z_train=Z_train,
            node_to_train_idx=node_to_train_idx,
            mu1=0.25,
            mu2=0.25,
            mu3=0.20,
            mu4=0.20,
            mu5=0.10,
            boundary_ratio=0.5,
            graph_region_mix=0.5,
            use_graph_association=True,
        )

        for term_name, term_val in terms.items():
            assert 0.0 <= term_val <= 1.0, f"Term '{term_name}' out of [0, 1] range: {term_val}"


# Verify pruning enforces at least 1 box per class even if min_boxes_per_class=0 and matches in fast_mode.
def test_prune_never_removes_last_class_box_and_fast_mode(pruning_setup: dict) -> None:
    p = pruning_setup
    cfg_zero = copy.deepcopy(p["cfg"])
    cfg_zero["min_boxes_per_class"] = 0
    cfg_zero["recall_safeguard"] = False
    pruned_safe, _ = gshr_prune(
        p["fmnn"], p["ctx"], cfg_zero,
        p["Z_train"], p["y_train"], p["train_ids"],
        p["Z_val"], p["y_val"],
    )
    assert pruned_safe.num_boxes() >= p["fmnn"].num_classes
    assert all(len(pruned_safe.boxes_of_class(c)) >= 1 for c in range(p["fmnn"].num_classes))

    cfg_std = copy.deepcopy(p["cfg"])
    cfg_fast = {**cfg_std, "fast_mode": True}
    std_fmnn, std_log = gshr_prune(
        p["fmnn"], p["ctx"], cfg_std,
        p["Z_train"], p["y_train"], p["train_ids"],
        p["Z_val"], p["y_val"],
    )
    fast_fmnn, fast_log = gshr_prune(
        p["fmnn"], p["ctx"], cfg_fast,
        p["Z_train"], p["y_train"], p["train_ids"],
        p["Z_val"], p["y_val"],
    )
    assert fast_fmnn.num_boxes() == std_fmnn.num_boxes()
    assert sorted(fast_fmnn.boxes.keys()) == sorted(std_fmnn.boxes.keys())
    assert fast_log.val_macro_f1_after == pytest.approx(std_log.val_macro_f1_after)


# Verify pure-NumPy confusion metrics match sklearn macro-F1 and per-class recall identically.
def test_fast_confusion_metrics_matches_sklearn() -> None:
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    rng = np.random.RandomState(123)
    for num_classes, y_true, preds in [
        (5, rng.randint(0, 5, size=200), rng.randint(0, 5, size=200)),
        (4, np.array([0, 0, 2, 2, 2]), np.array([0, 2, 2, 1, 2])),  # class 3 absent everywhere
    ]:
        acc, macro_f1, recalls = _fast_confusion_metrics(y_true, preds, num_classes)
        sk_acc = float(accuracy_score(y_true, preds))
        sk_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
        sk_rec = recall_score(y_true, preds, average=None, labels=range(num_classes), zero_division=0)
        assert acc == pytest.approx(sk_acc, abs=1e-14)
        assert macro_f1 == pytest.approx(sk_f1, abs=1e-14)
        assert np.allclose(recalls, sk_rec, atol=1e-14)
