"""Unit tests for utility functions."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.utils import Timer, get_device, load_config, load_json, save_json, set_seed


def test_set_seed_reproducibility() -> None:
    """Verify that setting the seed produces deterministic random numbers."""
    set_seed(123)
    np_rand1 = np.random.rand(5)
    torch_rand1 = torch.rand(5)

    set_seed(123)
    np_rand2 = np.random.rand(5)
    torch_rand2 = torch.rand(5)

    assert np.allclose(np_rand1, np_rand2)
    assert torch.allclose(torch_rand1, torch_rand2)


def test_load_config() -> None:
    """Verify loading default configuration file."""
    cfg = load_config("configs/default.yaml")
    assert "data" in cfg
    assert cfg["data"]["name"] == "Cora"
    assert "gnn" in cfg
    assert "fmnn" in cfg
    assert "merge" in cfg
    assert "prune" in cfg
    assert "run" in cfg
    assert "safeguards" in cfg


def test_get_device() -> None:
    """Verify device selection."""
    cfg = {"run": {"device": "cpu"}}
    device = get_device(cfg)
    assert device == torch.device("cpu")


def test_json_io(tmp_path: Path) -> None:
    """Verify save_json and load_json with various datatypes."""
    test_file = tmp_path / "sub" / "test.json"
    data = {"int": 1, "float": 2.5, "list": [1, 2, 3], "np_int": np.int64(42)}
    save_json(data, test_file)
    loaded = load_json(test_file)
    assert loaded["int"] == 1
    assert loaded["float"] == 2.5
    assert loaded["list"] == [1, 2, 3]
    assert loaded["np_int"] == 42


def test_timer() -> None:
    """Verify Timer context manager records time into a dictionary."""
    record = {}
    with Timer(record, "test_task"):
        time.sleep(0.05)
    assert "test_task" in record
    assert record["test_task"] >= 0.04
