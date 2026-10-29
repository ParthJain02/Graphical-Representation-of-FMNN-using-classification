import numpy as np

from src.data import load_cora_like_dataset


def test_split_respects_label_leakage_guardrails():
    ds = load_cora_like_dataset(num_nodes=12, num_features=4, num_classes=3, seed=3)
    train_ids = np.where(ds.train_mask)[0]
    val_ids = np.where(ds.val_mask)[0]
    test_ids = np.where(ds.test_mask)[0]
    assert set(train_ids).isdisjoint(set(val_ids))
    assert set(train_ids).isdisjoint(set(test_ids))
    assert set(val_ids).isdisjoint(set(test_ids))
    assert train_ids.size + val_ids.size + test_ids.size == len(ds.y)
