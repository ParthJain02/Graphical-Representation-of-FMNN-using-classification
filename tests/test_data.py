import numpy as np

from src.data import load_cora_like_dataset, load_dataset, validate_masks


def test_dataset_masks_are_valid():
    ds = load_cora_like_dataset(num_nodes=18, num_features=5, num_classes=3, seed=7)
    validate_masks(ds.y, ds.train_mask, ds.val_mask, ds.test_mask)
    assert ds.x.shape == (18, 5)
    assert np.isfinite(ds.x).all()
    assert ds.num_classes == 3


def test_split_is_disjoint_and_exhaustive():
    ds = load_cora_like_dataset(num_nodes=20, num_features=6, num_classes=4, seed=11)
    assert np.sum(ds.train_mask) > 0
    assert np.sum(ds.val_mask) > 0
    assert np.sum(ds.test_mask) > 0
    assert not (ds.train_mask & ds.val_mask).any()
    assert not (ds.train_mask & ds.test_mask).any()
    assert not (ds.val_mask & ds.test_mask).any()


def test_real_cora_loader_works():
    ds = load_dataset('Cora', root='data/test-cora')
    assert ds.x.shape[0] == ds.y.shape[0]
    assert ds.x.shape[1] > 0
    assert ds.num_classes == 7
    assert ds.train_mask.sum() > 0
    assert ds.val_mask.sum() > 0
    assert ds.test_mask.sum() > 0
