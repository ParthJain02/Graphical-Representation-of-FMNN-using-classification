"""Compute and display summary graph statistics of base FMNN hyperboxes on Cora."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import load_dataset
from src.embeddings import extract_embeddings, fit_transform_all
from src.fmnn import FMNN
from src.gnn import train_gnn
from src.graph_stats import BoxGraphCache
from src.utils import load_config


def main() -> None:
    cfg = load_config("configs/default.yaml")
    data = load_dataset("Cora", "./data")
    model, _, _ = train_gnn(data, cfg, seed=0)

    Z = extract_embeddings(model, data)
    Z_scaled = fit_transform_all(Z, data.train_idx, clip=True)

    train_idx = data.train_idx.cpu().numpy()
    Z_train = Z_scaled[train_idx]
    y_train = data.y[data.train_idx].cpu().numpy()
    node_ids = train_idx.tolist()

    fmnn = FMNN(theta=cfg["fmnn"]["theta"], gamma=cfg["fmnn"]["gamma"])
    fmnn.fit(Z_train, y_train, node_ids)

    # Boolean train mask
    train_mask = np.zeros(data.num_nodes, dtype=bool)
    train_mask[train_idx] = True

    cache = BoxGraphCache(
        adj_csr=data.adj_csr,
        degree=data.degree,
        y=data.y.cpu().numpy(),
        train_mask=train_mask,
    )
    cache.build_all(fmnn)

    purities = []
    conductances = []
    agreements = []
    single_member_count = 0
    total_boxes = fmnn.num_boxes()

    for bid, box in sorted(fmnn.boxes.items()):
        stats = cache.get(bid)
        assert stats is not None
        purities.append(stats.purity)
        conductances.append(stats.conductance)
        agreements.append(stats.neighbor_agreement)
        if len(box.members) == 1:
            single_member_count += 1

    print("=" * 65)
    print(f"Cora Base FMNN Graph Statistics Summary (theta={cfg['fmnn']['theta']}, seed=0)")
    print("=" * 65)
    print(f"  Total Hyperboxes:              {total_boxes}")
    print(f"  Boxes with only 1 member:      {single_member_count} / {total_boxes} ({single_member_count/total_boxes*100:.1f}%)")
    print(f"  Class Purity:                  mean={np.mean(purities):.4f}, min={np.min(purities):.4f}, max={np.max(purities):.4f}")
    print(f"  Conductance:                   mean={np.mean(conductances):.4f}, min={np.min(conductances):.4f}, max={np.max(conductances):.4f}")
    print(f"  Neighbor Label Agreement:      mean={np.mean(agreements):.4f}, min={np.min(agreements):.4f}, max={np.max(agreements):.4f}")
    print("=" * 65)
    print("\nPer-box breakdown:")
    print(f"{'Box ID':>6} | {'Class':>5} | {'Members':>7} | {'Internal':>8} | {'Boundary':>8} | {'Volume':>6} | {'Conduct':>7} | {'Purity':>6} | {'Agreem':>6}")
    print("-" * 75)
    for bid, box in sorted(fmnn.boxes.items()):
        s = cache.get(bid)
        assert s is not None
        print(f"{bid:6d} | {box.cls:5d} | {len(box.members):7d} | {s.internal_edges:8d} | {s.boundary_edges:8d} | {s.volume:6d} | {s.conductance:7.4f} | {s.purity:6.2f} | {s.neighbor_agreement:6.2f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
