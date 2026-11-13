"""Shared helpers for CLI wrappers in scripts/."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import GraphData, load_dataset
from src.pipeline import prepare_seed_context, run_single_seed
from src.utils import load_config


SUMMARY_METRIC_KEYS = [
    "test_accuracy",
    "test_macro_f1",
    "test_micro_f1",
    "val_accuracy",
    "val_macro_f1",
    "num_boxes",
    "mean_box_size",
    "max_box_size",
    "cross_class_overlap",
    "gnn_train_time",
    "fmnn_fit_time",
    "refine_time",
    "pred_latency_per_node",
    "peak_memory_mb",
]


def load_cfg_and_data(config_path: str) -> tuple[dict[str, Any], GraphData]:
    """Load YAML configuration and dataset specified under data.name."""
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "Cora")
    data_root = data_cfg.get("root", "./data")
    data = load_dataset(dataset_name, root=data_root)
    return cfg, data


def get_seeds(cfg: dict[str, Any], seeds_override: list[int] | None = None) -> list[int]:
    """Resolve list of seeds from CLI override or config."""
    if seeds_override is not None:
        return [int(s) for s in seeds_override]
    return [int(s) for s in cfg.get("run", {}).get("seeds", [0, 1, 2, 3, 4])]


def get_tables_dir(cfg: dict[str, Any]) -> Path:
    """Return results/tables directory path derived from run.output_dir."""
    out_root = Path(cfg.get("run", {}).get("output_dir", "./results"))
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir


def execute_variants(
    cfg: dict[str, Any],
    data: GraphData,
    seeds: list[int],
    variants: list[str],
) -> list[dict[str, Any]]:
    """Run one or more variants across seeds, reusing GNN and base FMNN per seed."""
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_ctx = prepare_seed_context(cfg, data, seed)
        for variant in variants:
            res = run_single_seed(cfg, data, seed, variant, seed_ctx=seed_ctx)
            runs.append(res)
    return runs


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate mean and sample std across seeds for each variant."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(row["variant"], []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for variant_name, group in grouped.items():
        entry: dict[str, Any] = {"num_seeds": len(group)}
        for key in SUMMARY_METRIC_KEYS:
            vals = [float(r[key]) for r in group if r.get(key) is not None]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            entry[key] = {"mean": mean_val, "std": std_val}
        summary[variant_name] = entry
    return summary
