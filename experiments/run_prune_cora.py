"""Run and evaluate graph-aware hyperbox pruning on real Cora embeddings (seed 0)."""

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
from src.merge import gshr_merge
from src.prune import gshr_prune
from src.utils import load_config


def main() -> None:
    cfg = load_config("configs/default.yaml")
    data = load_dataset("Cora", "./data")

    # 1. Train GNN with seed 0
    model, _, _ = train_gnn(data, cfg, seed=0)

    # 2. Extract and scale embeddings
    Z = extract_embeddings(model, data)
    Z_scaled = fit_transform_all(Z, data.train_idx, clip=True)

    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()

    Z_train = Z_scaled[train_idx]
    y_train = data.y[data.train_idx].cpu().numpy()
    train_ids = train_idx.tolist()

    Z_val = Z_scaled[val_idx]
    y_val = data.y[data.val_idx].cpu().numpy()

    # 3. Fit base FMNN
    fmnn = FMNN(theta=cfg["fmnn"]["theta"], gamma=cfg["fmnn"]["gamma"])
    fmnn.fit(Z_train, y_train, train_ids)

    # 4. Merge step
    merge_cfg = dict(cfg["merge"])
    merge_cfg["theta"] = cfg["fmnn"]["theta"]
    merge_cfg["safeguards"] = cfg["safeguards"]
    merged_fmnn, merge_log = gshr_merge(fmnn, data, merge_cfg)

    # 5. Pruning step
    prune_cfg = dict(cfg["prune"])
    prune_cfg["safeguards"] = cfg["safeguards"]
    pruned_fmnn, prune_log = gshr_prune(
        fmnn=merged_fmnn,
        graph_ctx=data,
        cfg_prune=prune_cfg,
        Z_train=Z_train,
        y_train=y_train,
        train_ids=train_ids,
        Z_val=Z_val,
        y_val=y_val,
    )

    print("=" * 65)
    print("GSHR Pruning Results on Cora (Seed 0)")
    print("=" * 65)
    print(f"  Boxes Before Pruning (after merge):  {prune_log.boxes_before}")
    print(f"  Boxes After Pruning:                 {prune_log.boxes_after}")
    print(f"  Total Boxes Pruned:                  {prune_log.pruned_count}")
    print(f"  Elapsed Prune Time:                  {prune_log.elapsed_time:.4f} s")
    print("-" * 65)
    print(f"  Validation Acc (Before Prune):       {prune_log.val_acc_before:.4f}")
    print(f"  Validation Acc (After Prune):        {prune_log.val_acc_after:.4f} (diff: {prune_log.val_acc_after - prune_log.val_acc_before:+.4f})")
    print(f"  Validation Macro-F1 (Before Prune):  {prune_log.val_macro_f1_before:.4f}")
    print(f"  Validation Macro-F1 (After Prune):   {prune_log.val_macro_f1_after:.4f} (diff: {prune_log.val_macro_f1_after - prune_log.val_macro_f1_before:+.4f})")
    print("-" * 65)
    print("  Validation Per-Class Recall Change:")
    print(f"  {'Class':>5} | {'Recall Before':>13} | {'Recall After':>12} | {'Change':>8}")
    print("  " + "-" * 48)
    for c in range(data.num_classes):
        r_bef = prune_log.val_recalls_before[c]
        r_aft = prune_log.val_recalls_after[c]
        diff = r_aft - r_bef
        print(f"  {c:5d} | {r_bef:13.4f} | {r_aft:12.4f} | {diff:+8.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
