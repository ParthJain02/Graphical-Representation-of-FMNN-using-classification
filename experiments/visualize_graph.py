"""Graph and Hyperbox Representation Visualizer.

Generates visual representations of:
1. Cora Citation Graph Topology (NetworkX network structure with nodes and edges)
2. GNN Embedding Manifold (2D PCA projection of 64D node embeddings)
3. Base FMNN Hyperboxes before refinement (showing box fragmentation)
4. GSHR Refined Hyperboxes after graph-aware merging and pruning (showing compression and purity)

Saves the combined visualization to results/figures/graph_representation.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

from src.data import load_dataset
from src.merge import GraphContext
from src.pipeline import prepare_seed_context, select_hyperparameters
from src.utils import load_config


# Standard color palette for 7 Cora classes
CORA_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
]

CORA_CLASS_NAMES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def run_visualization() -> None:
    print("=" * 75)
    print("GENERATING GRAPH AND HYPERBOX REPRESENTATION VISUALIZATION")
    print("=" * 75)

    cfg_path = ROOT_DIR / "configs" / "default.yaml"
    cfg = load_config(str(cfg_path))

    # 1. Load Cora Dataset
    print("\n[Step 1] Loading Cora dataset...")
    data = load_dataset("Cora", root=str(ROOT_DIR / "data"))
    print(f"  Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1] // 2}, Classes: {data.num_classes}")

    # Build boolean masks
    train_mask = np.zeros(data.num_nodes, dtype=bool)
    train_mask[data.train_idx.cpu().numpy()] = True

    val_mask = np.zeros(data.num_nodes, dtype=bool)
    val_mask[data.val_idx.cpu().numpy()] = True

    test_mask = np.zeros(data.num_nodes, dtype=bool)
    test_mask[data.test_idx.cpu().numpy()] = True

    # 2. Extract Seed 0 Context (GNN, Embeddings, Base FMNN)
    print("\n[Step 2] Training GNN & fitting Base FMNN on Seed 0...")
    seed = 0
    ctx = prepare_seed_context(cfg, data, seed=seed)
    base_fmnn = ctx.base_fmnn
    Z_scaled = ctx.Z_scaled  # (2708, 64)

    train_idx = data.train_idx.cpu().numpy()
    val_idx = data.val_idx.cpu().numpy()
    test_idx = data.test_idx.cpu().numpy()
    y_all = data.y.cpu().numpy()

    Z_train = Z_scaled[train_idx]
    y_train = y_all[train_idx]
    train_ids = train_idx.tolist()

    Z_val = Z_scaled[val_idx]
    y_val = y_all[val_idx]

    graph_ctx = GraphContext(
        adj_csr=data.adj_csr,
        degree=data.degree,
        y=y_all,
        train_mask=train_mask,
    )

    # 3. Refine FMNN with GSHR (B6: Merge + Prune)
    print("\n[Step 3] Applying GSHR Graph-Aware Merging and Pruning...")
    gshr_fmnn, best_tm, best_tp, _ = select_hyperparameters(
        base_fmnn=base_fmnn,
        graph_ctx=graph_ctx,
        cfg=cfg,
        Z_train=Z_train,
        y_train=y_train,
        train_ids=train_ids,
        Z_val=Z_val,
        y_val=y_val,
        variant="B6",
        base_val_macro_f1=ctx.base_val_macro_f1,
    )

    print(f"  Base FMNN Boxes: {base_fmnn.num_boxes()}")
    print(f"  GSHR Refined Boxes: {gshr_fmnn.num_boxes()} (Reduction: {(1 - gshr_fmnn.num_boxes()/base_fmnn.num_boxes())*100:.1f}%)")

    # 4. Dimensionality Reduction (PCA on Training Embeddings)
    print("\n[Step 4] Projecting 64D embeddings to 2D for visualization...")
    pca = PCA(n_components=2, random_state=seed)
    pca.fit(Z_train)
    Z_2d = pca.transform(Z_scaled)

    # 5. Extract Network Subgraph for Graph Representation
    print("\n[Step 5] Building Graph Topology Representation (NetworkX)...")
    G = nx.Graph()
    # Add training nodes
    for u in train_idx:
        G.add_node(int(u), label=int(y_all[u]), is_train=True)

    # Add edges between training nodes or their immediate 1-hop connected neighbors
    edge_index = data.edge_index.cpu().numpy()
    train_set = set(train_idx)
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < v and (u in train_set and v in train_set):
            G.add_edge(u, v)

    # If density is low, add 1-hop interconnects
    if G.number_of_edges() < 30:
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u < v and (u in train_set or v in train_set):
                if u not in G:
                    G.add_node(u, label=int(y_all[u]), is_train=False)
                if v not in G:
                    G.add_node(v, label=int(y_all[v]), is_train=False)
                G.add_edge(u, v)

    # Subgraph for visualization (largest component or core nodes)
    sub_nodes = [n for n in G.nodes() if G.degree(n) > 0]
    if len(sub_nodes) > 160:
        sub_nodes = sorted(sub_nodes, key=lambda n: G.degree(n), reverse=True)[:150]
    sub_G = G.subgraph(sub_nodes)

    pos = nx.spring_layout(sub_G, seed=42, k=0.28, iterations=60)

    # 6. Plot Comprehensive 4-Panel Visualization
    print("\n[Step 6] Rendering 4-Panel Publication-Quality Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    plt.subplots_adjust(wspace=0.22, hspace=0.25)

    # -------------------------------------------------------------
    # Panel 1: Graph Topology & Community Connectivity
    # -------------------------------------------------------------
    ax_graph = axes[0, 0]
    ax_graph.set_title("A. Cora Graph Topology & Citation Network Structure", fontsize=14, fontweight="bold", pad=12)

    # Draw edges
    nx.draw_networkx_edges(sub_G, pos, ax=ax_graph, alpha=0.35, edge_color="#7f8c8d", width=1.0)

    # Draw nodes colored by class
    node_colors = [CORA_COLORS[sub_G.nodes[n]["label"] % len(CORA_COLORS)] for n in sub_G.nodes()]
    node_sizes = [45 + 18 * sub_G.degree(n) for n in sub_G.nodes()]
    nx.draw_networkx_nodes(sub_G, pos, ax=ax_graph, node_color=node_colors, node_size=node_sizes, edgecolors="#2c3e50", linewidths=0.8)

    ax_graph.axis("off")
    ax_graph.text(0.02, 0.02, f"Visualized Subgraph: {sub_G.number_of_nodes()} nodes, {sub_G.number_of_edges()} edges\nNodes sized by degree; colored by class",
                  transform=ax_graph.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7", alpha=0.9))

    # -------------------------------------------------------------
    # Panel 2: GNN Embedding Manifold (PCA 2D)
    # -------------------------------------------------------------
    ax_embed = axes[0, 1]
    ax_embed.set_title("B. Learned GNN Node Embeddings (2D PCA Projection)", fontsize=14, fontweight="bold", pad=12)

    for c in range(data.num_classes):
        c_mask = (y_all == c)
        c_train = c_mask & train_mask
        c_test = c_mask & test_mask

        ax_embed.scatter(
            Z_2d[c_test, 0], Z_2d[c_test, 1],
            c=CORA_COLORS[c % len(CORA_COLORS)],
            alpha=0.25, s=20, marker="o", edgecolors="none"
        )
        ax_embed.scatter(
            Z_2d[c_train, 0], Z_2d[c_train, 1],
            c=CORA_COLORS[c % len(CORA_COLORS)],
            alpha=0.95, s=65, marker="^", edgecolors="black", linewidths=0.9,
            label=f"{CORA_CLASS_NAMES[c]}"
        )

    ax_embed.set_xlabel("Principal Component 1", fontsize=11)
    ax_embed.set_ylabel("Principal Component 2", fontsize=11)
    ax_embed.grid(True, linestyle=":", alpha=0.5)
    ax_embed.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax_embed.text(0.02, 0.02, "Triangles = Training Nodes (N=140)\nFaint Dots = Unlabeled / Test Nodes",
                  transform=ax_embed.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7", alpha=0.9))

    # Helper function to project hyperbox boundaries
    def get_projected_box_rect(box, member_points_2d):
        if len(member_points_2d) > 0:
            min_pt = np.min(member_points_2d, axis=0)
            max_pt = np.max(member_points_2d, axis=0)
            # Add small padding for single points
            padding = 0.03
            w = max(max_pt[0] - min_pt[0], padding)
            h = max(max_pt[1] - min_pt[1], padding)
            return min_pt[0] - padding/2, min_pt[1] - padding/2, w + padding, h + padding
        else:
            v_2d = pca.transform(box.v.reshape(1, -1))[0]
            w_2d = pca.transform(box.w.reshape(1, -1))[0]
            x = min(v_2d[0], w_2d[0])
            y = min(v_2d[1], w_2d[1])
            return x, y, max(abs(w_2d[0] - v_2d[0]), 0.04), max(abs(w_2d[1] - v_2d[1]), 0.04)

    # -------------------------------------------------------------
    # Panel 3: Base FMNN Hyperboxes (Before Refinement)
    # -------------------------------------------------------------
    ax_base = axes[1, 0]
    ax_base.set_title(f"C. Base FMNN Hyperboxes (Before Refinement: {base_fmnn.num_boxes()} Boxes)", fontsize=14, fontweight="bold", pad=12)

    # Plot train nodes
    for c in range(data.num_classes):
        c_train = (y_all == c) & train_mask
        ax_base.scatter(
            Z_2d[c_train, 0], Z_2d[c_train, 1],
            c=CORA_COLORS[c % len(CORA_COLORS)],
            alpha=0.85, s=55, marker="o", edgecolors="black", linewidths=0.7
        )

    # Draw Base FMNN boxes
    for b_id, box in base_fmnn.boxes.items():
        box_members = [m for m in box.members if m < len(Z_scaled)]
        pts_2d = Z_2d[box_members] if box_members else np.empty((0, 2))
        x, y, w, h = get_projected_box_rect(box, pts_2d)
        rect = Rectangle(
            (x, y), w, h,
            linewidth=1.8, edgecolor=CORA_COLORS[box.cls % len(CORA_COLORS)],
            facecolor=CORA_COLORS[box.cls % len(CORA_COLORS)], alpha=0.15,
            linestyle="--"
        )
        ax_base.add_patch(rect)
        ax_base.text(x + w/2, y + h/2, f"B{box.id}",
                     color=CORA_COLORS[box.cls % len(CORA_COLORS)],
                     fontsize=8.5, fontweight="bold", ha="center", va="center")

    ax_base.set_xlabel("Principal Component 1", fontsize=11)
    ax_base.set_ylabel("Principal Component 2", fontsize=11)
    ax_base.grid(True, linestyle=":", alpha=0.5)
    ax_base.text(0.02, 0.02, f"Base Model: {base_fmnn.num_boxes()} Hyperboxes\nRedundant, fragmented boundaries",
                 transform=ax_base.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7", alpha=0.9))

    # -------------------------------------------------------------
    # Panel 4: GSHR Refined Hyperboxes (After Merge & Prune)
    # -------------------------------------------------------------
    ax_gshr = axes[1, 1]
    ax_gshr.set_title(f"D. GSHR Refined Hyperboxes (Merged & Pruned: {gshr_fmnn.num_boxes()} Boxes)", fontsize=14, fontweight="bold", pad=12)

    # Plot train nodes
    for c in range(data.num_classes):
        c_train = (y_all == c) & train_mask
        ax_gshr.scatter(
            Z_2d[c_train, 0], Z_2d[c_train, 1],
            c=CORA_COLORS[c % len(CORA_COLORS)],
            alpha=0.85, s=55, marker="o", edgecolors="black", linewidths=0.7
        )

    # Draw GSHR boxes
    for b_id, box in gshr_fmnn.boxes.items():
        box_members = [m for m in box.members if m < len(Z_scaled)]
        pts_2d = Z_2d[box_members] if box_members else np.empty((0, 2))
        x, y, w, h = get_projected_box_rect(box, pts_2d)
        rect = Rectangle(
            (x, y), w, h,
            linewidth=2.4, edgecolor=CORA_COLORS[box.cls % len(CORA_COLORS)],
            facecolor=CORA_COLORS[box.cls % len(CORA_COLORS)], alpha=0.22,
            linestyle="-"
        )
        ax_gshr.add_patch(rect)
        ax_gshr.text(x + w/2, y + h/2, f"B{box.id}\n(N={len(box.members)})",
                     color=CORA_COLORS[box.cls % len(CORA_COLORS)],
                     fontsize=9, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=CORA_COLORS[box.cls % len(CORA_COLORS)], alpha=0.85))

    ax_gshr.set_xlabel("Principal Component 1", fontsize=11)
    ax_gshr.set_ylabel("Principal Component 2", fontsize=11)
    ax_gshr.grid(True, linestyle=":", alpha=0.5)
    ax_gshr.text(0.02, 0.02, f"GSHR Model: {gshr_fmnn.num_boxes()} Hyperboxes\nMerged along graph edges & pruned",
                 transform=ax_gshr.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7", alpha=0.9))

    # Ensure both bottom panels share axis limits for direct comparison
    x_min = min(ax_base.get_xlim()[0], ax_gshr.get_xlim()[0])
    x_max = max(ax_base.get_xlim()[1], ax_gshr.get_xlim()[1])
    y_min = min(ax_base.get_ylim()[0], ax_gshr.get_ylim()[0])
    y_max = max(ax_base.get_ylim()[1], ax_gshr.get_ylim()[1])
    ax_base.set_xlim(x_min, x_max)
    ax_base.set_ylim(y_min, y_max)
    ax_gshr.set_xlim(x_min, x_max)
    ax_gshr.set_ylim(y_min, y_max)

    # Save outputs
    figures_dir = ROOT_DIR / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig_path1 = figures_dir / "graph_representation.png"
    fig_path2 = figures_dir / "graph_hyperbox_visualization.png"

    plt.tight_layout()
    plt.savefig(str(fig_path1), dpi=300)
    plt.savefig(str(fig_path2), dpi=300)
    plt.close()

    print(f"\n[Done] Successfully saved graph visualization to:")
    print(f"  -> {fig_path1}")
    print(f"  -> {fig_path2}")
    print("=" * 75)


if __name__ == "__main__":
    run_visualization()
