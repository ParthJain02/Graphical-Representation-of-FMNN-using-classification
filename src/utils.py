"""General utility functions and helpers for GSHR."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set seeds for python random, numpy, and PyTorch for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """Load configuration dictionary from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_device(cfg: dict[str, Any] | None = None) -> torch.device:
    """Determine the compute device (CPU or CUDA) from config or system availability."""
    device_setting = "auto"
    if cfg is not None:
        if "run" in cfg and "device" in cfg["run"]:
            device_setting = cfg["run"]["device"]
        elif "device" in cfg:
            device_setting = cfg["device"]

    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def _json_default(obj: Any) -> Any:
    """JSON serializer helper for numpy types and paths."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path, torch.device)):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(obj: Any, path: str | Path) -> None:
    """Save an object as a JSON file, creating parent directories if needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    """Load an object from a JSON file."""
    file_path = Path(path)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    """Context manager to record elapsed execution time in seconds into a dictionary."""

    def __init__(self, record_dict: dict[str, float] | None = None, key: str | None = None) -> None:
        self.record_dict = record_dict
        self.key = key
        self.elapsed: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> Timer:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed = time.perf_counter() - self._start_time
        if self.record_dict is not None and self.key is not None:
            self.record_dict[self.key] = self.elapsed
