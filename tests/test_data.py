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


def test_planetoid_loader_canonicalizes_citeseer(monkeypatch):
    import src.data as data_module

    calls = {}

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return self.value

        def max(self):
            return self

        def item(self):
            return 1

    class FakeData:
        x = FakeTensor(np.ones((2, 3), dtype=np.float32))
        y = FakeTensor(np.array([0, 1], dtype=np.int64))
        edge_index = FakeTensor(np.array([[0, 1], [1, 0]], dtype=np.int64))
        train_mask = FakeTensor(np.array([True, False]))
        val_mask = FakeTensor(np.array([False, True]))
        test_mask = FakeTensor(np.array([False, False]))

    class FakePlanetoid:
        def __init__(self, root, name, split):
            calls.update(root=root, name=name, split=split)

        def __getitem__(self, index):
            return FakeData()

    monkeypatch.setattr(data_module, 'Planetoid', FakePlanetoid)
    dataset = data_module.load_dataset('citeseer')
    assert calls['name'] == 'CiteSeer'
    assert dataset.metadata['dataset'] == 'CiteSeer'
