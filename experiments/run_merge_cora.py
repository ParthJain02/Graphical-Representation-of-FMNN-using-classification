"""Run and evaluate graph-aware hyperbox merging on real Cora embeddings (seed 0)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import f1_score

from src.data import load_dataset
from src.embeddings import extract_embeddings, fit_transform_all
from src.fmnn import FMNN
from src.gnn import train_gnn
from src.merge import gshr_merge
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
    node_ids = train_idx.tolist()

    Z_val = Z_scaled[val_idx]
    y_val = data.y[data.val_idx].cpu().numpy()

    # 3. Fit base FMNN
    fmnn = FMNN(theta=cfg["fmnn"]["theta"], gamma=cfg["fmnn"]["gamma"])
    fmnn.fit(Z_train, y_train, node_ids)

    # Validation performance BEFORE merge
    val_preds_before = fmnn.predict(Z_val)
    val_acc_before = float((val_preds_before == y_val).mean())
    val_f1_before = float(f1_score(y_val, val_preds_before, average="macro", zero_division=0))

    # 4. Run GSHR Merge
    merge_cfg = dict(cfg["merge"])
    merge_cfg["theta"] = cfg["fmnn"]["theta"]
    merge_cfg["safeguards"] = cfg["safeguards"]

    merged_fmnn, merge_log = gshr_merge(fmnn, data, merge_cfg)

    # Validation performance AFTER merge
    val_preds_after = merged_fmnn.predict(Z_val)
    val_acc_after = float((val_preds_after == y_val).mean())
    val_f1_after = float(f1_score(y_val, val_preds_after, average="macro", zero_division=0))

    print("=" * 65)
    print("GSHR Merging Results on Cora (Seed 0)")
    print("=" * 65)
    print(f"  Boxes Before Merge:            {merge_log.boxes_before}")
    print(f"  Boxes After Merge:             {merge_log.boxes_after}")
    print(f"  Total Merges Executed:         {len(merge_log)}")
    print(f"  Rejection Reason Counts:")
    for reason, count in sorted(merge_log.rejections.items()):
        print(f"    - {reason:<22}: {count}")
    print("-" * 65)
    print(f"  Validation Acc (Before):       {val_acc_before:.4f}")
    print(f"  Validation Acc (After):        {val_acc_after:.4f} (diff: {val_acc_after - val_acc_before:+.4f})")
    print(f"  Validation Macro-F1 (Before):  {val_f1_before:.4f}")
    print(f"  Validation Macro-F1 (After):   {val_f1_after:.4f} (diff: {val_f1_after - val_f1_before:+.4f})")
    print(f"  Elapsed Merge Time:            {merge_log.elapsed_time:.4f} s")
    print("=" * 65)


if __name__ == "__main__":
    main()
