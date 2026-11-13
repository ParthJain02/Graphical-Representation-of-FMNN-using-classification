"""Experiment script to scan FMNN theta values on Cora validation split."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.metrics import f1_score

from src.data import load_dataset
from src.embeddings import extract_embeddings, fit_transform_all
from src.fmnn import FMNN
from src.gnn import train_gnn
from src.utils import load_config


def main() -> None:
    cfg = load_config("configs/default.yaml")
    data = load_dataset("Cora", "./data")
    model, _, _ = train_gnn(data, cfg, seed=0)

    Z = extract_embeddings(model, data)
    Z_scaled = fit_transform_all(Z, data.train_idx, clip=True)

    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()

    Z_train = Z_scaled[train_idx]
    y_train = data.y[data.train_idx].cpu().numpy()
    node_ids = train_idx.tolist()

    Z_val = Z_scaled[val_idx]
    y_val = data.y[data.val_idx].cpu().numpy()

    results = []
    thetas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print(f"{'theta':>8} | {'boxes':>8} | {'train_acc':>10} | {'val_acc':>10} | {'val_macro_f1':>12}")
    print("-" * 60)

    for th in thetas:
        fmnn = FMNN(theta=th, gamma=cfg["fmnn"]["gamma"])
        fmnn.fit(Z_train, y_train, node_ids)

        train_preds = fmnn.predict(Z_train)
        train_acc = float((train_preds == y_train).mean())

        val_preds = fmnn.predict(Z_val)
        val_acc = float((val_preds == y_val).mean())
        val_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))

        nb = fmnn.num_boxes()
        print(f"{th:8.1f} | {nb:8d} | {train_acc:10.4f} | {val_acc:10.4f} | {val_f1:12.4f}")
        results.append({
            "theta": th,
            "num_boxes": nb,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

    Path("results").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/fmnn_theta_scan.csv", index=False)
    print("\nSaved table to results/fmnn_theta_scan.csv")


if __name__ == "__main__":
    main()
