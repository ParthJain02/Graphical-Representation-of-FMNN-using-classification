"""
Generate visually stunning animated GIFs for GitHub README.
Visualizes the complete lifecycle:
  1. GNN Embedding space with graph topological edges
  2. FMNN Hyperbox Expansion & Simpson Contraction (Fragmented boxes)
  3. GSHR Graph-Structure-Aware Pruning (Dissolving redundant boxes)
  4. GSHR Same-Class Safe Merging (Consolidation)
  5. Hybrid Gated Fuzzy Classification & Decision Contours
"""

import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

OUT_DIR = Path("report_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GIF_PATH = OUT_DIR / "gshr_refinement_lifecycle.gif"

# Set deterministic seed
np.random.seed(42)

# Palette (modern GitHub dark theme aesthetic)
BG_COLOR = "#0b0f19"
CARD_BG = "#151c2e"
GRID_COLOR = "#1f293d"
TEXT_COLOR = "#f0f6fc"
TEXT_MUTED = "#8b949e"
EDGE_COLOR = "#2d3748"

# 4 class colors: Neon Blue, Coral Pink, Vibrant Green, Electric Purple
CLASS_COLORS = ["#00f0ff", "#ff0055", "#00ff66", "#b026ff"]
CLASS_NAMES = ["Neural Nets (C0)", "Genetic Alg (C1)", "Prob Methods (C2)", "Theory (C3)"]

# Generate 4 synthetic clusters in [0.1, 0.9]^2
centers = np.array([
    [0.28, 0.72],
    [0.72, 0.75],
    [0.30, 0.28],
    [0.74, 0.30],
])

n_per_class = 22
nodes = []
labels = []
train_mask = []

for c, center in enumerate(centers):
    pts = np.random.randn(n_per_class, 2) * 0.075 + center
    pts = np.clip(pts, 0.05, 0.95)
    nodes.append(pts)
    labels.extend([c] * n_per_class)
    # 5 train nodes per class, rest test
    t_m = [True] * 5 + [False] * (n_per_class - 5)
    np.random.shuffle(t_m)
    train_mask.extend(t_m)

nodes = np.vstack(nodes)
labels = np.array(labels)
train_mask = np.array(train_mask)
test_mask = ~train_mask

# Generate edges: intra-class (high homophily ~0.8) and a few inter-class edges
edges = []
for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        dist = np.linalg.norm(nodes[i] - nodes[j])
        if labels[i] == labels[j]:
            if dist < 0.16 and np.random.rand() < 0.75:
                edges.append((i, j))
        else:
            if dist < 0.18 and np.random.rand() < 0.08:
                edges.append((i, j))

# Define Hyperboxes across stages
# Initial FMNN creates ~14 boxes (3-4 per class)
init_boxes = [
    # Class 0
    {"box": [0.18, 0.62, 0.36, 0.82], "c": 0, "status": "keep"},
    {"box": [0.22, 0.58, 0.32, 0.69], "c": 0, "status": "merge"},
    {"box": [0.38, 0.70, 0.44, 0.77], "c": 0, "status": "prune"},
    # Class 1
    {"box": [0.63, 0.66, 0.83, 0.84], "c": 1, "status": "keep"},
    {"box": [0.68, 0.60, 0.77, 0.70], "c": 1, "status": "merge"},
    {"box": [0.58, 0.78, 0.65, 0.85], "c": 1, "status": "prune"},
    # Class 2
    {"box": [0.20, 0.18, 0.38, 0.38], "c": 2, "status": "keep"},
    {"box": [0.25, 0.35, 0.35, 0.44], "c": 2, "status": "merge"},
    {"box": [0.14, 0.28, 0.21, 0.35], "c": 2, "status": "prune"},
    # Class 3
    {"box": [0.65, 0.20, 0.84, 0.40], "c": 3, "status": "keep"},
    {"box": [0.60, 0.28, 0.71, 0.38], "c": 3, "status": "merge"},
    {"box": [0.78, 0.14, 0.86, 0.22], "c": 3, "status": "prune"},
]

# Post-merge boxes (merged with the main keep box)
merged_boxes = [
    {"box": [0.18, 0.58, 0.36, 0.82], "c": 0},
    {"box": [0.63, 0.60, 0.83, 0.84], "c": 1},
    {"box": [0.20, 0.18, 0.38, 0.44], "c": 2},
    {"box": [0.60, 0.20, 0.84, 0.40], "c": 3},
]

frames = []
total_frames = 36

for f in range(total_frames):
    fig, ax = plt.subplots(figsize=(8.5, 6.4), dpi=110)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(CARD_BG)

    # Grid & borders
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(1.5)

    # Phase assignment across 36 frames:
    # 0..6: Phase 1 (GNN Embeddings + Graph Edges)
    # 7..14: Phase 2 (FMNN Expansion: 12 Fragmented Boxes form)
    # 15..22: Phase 3 (GSHR Prune: Redundant boxes flash red & dissolve)
    # 23..29: Phase 4 (GSHR Merge: Same-class boxes consolidate safely)
    # 30..35: Phase 5 (Hybrid Gated Inference: Contours & test classification)

    if f <= 6:
        phase_title = "STAGE 1: GNN Latent Space & Graph Homophily"
        phase_sub = f"Deep GNN (APPNP/GCNII) maps raw features into normalized Z in [0,1]^d (h = 0.810)"
        phase_badge = "GNN EMBEDDINGS"
        badge_color = "#3b82f6"
    elif f <= 14:
        phase_title = "STAGE 2: Base FMNN Hyperbox Expansion (B2 Baseline)"
        prog = (f - 7) / 7.0
        phase_sub = f"Incremental L1-expansion (theta=0.3): 12 hyperboxes form (fragmented & redundant)"
        phase_badge = "FMNN EXPANSION (12 BOXES)"
        badge_color = "#f59e0b"
    elif f <= 22:
        phase_title = "STAGE 3: Graph-Aware Refinement -- Pruning (B5/B9)"
        phase_sub = "Low graph-conductance & redundant boundary boxes identified and pruned (-33% boxes)"
        phase_badge = "GSHR PRUNING ACTIVE"
        badge_color = "#ef4444"
    elif f <= 29:
        phase_title = "STAGE 4: Graph-Aware Refinement -- Fast Merge (B6/B9)"
        phase_sub = "Same-class boxes merged under volume inflation <= 0.5 & zero cross-class overlap"
        phase_badge = "COMPACT REFINED BOXES (4 BOXES)"
        badge_color = "#10b981"
    else:
        phase_title = "STAGE 5: Hybrid Gated Decision & Rule Transparency (B13/B14)"
        phase_sub = "Pure single-box membership (98.2% acc) + GNN softmax blending (0 flips on Cora)"
        phase_badge = "INTERPRETABLE HYBRID PIPELINE"
        badge_color = "#8b5cf6"

    # Draw graph edges
    edge_alpha = 0.35 if f <= 6 else 0.18
    for i, j in edges:
        ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                color=EDGE_COLOR, alpha=edge_alpha, linewidth=0.9, zorder=1)

    # Draw hyperboxes depending on phase
    if 7 <= f <= 14:
        # Expanding initial boxes
        growth = min(1.0, (f - 6) / 6.0)
        for b in init_boxes:
            v1, w1, v2, w2 = b["box"]
            cx = (v1 + v2) / 2
            cy = (w1 + w2) / 2
            dx = (v2 - v1) * growth / 2
            dy = (w2 - w1) * growth / 2
            col = CLASS_COLORS[b["c"]]
            rect = patches.Rectangle(
                (cx - dx, cy - dy), dx * 2, dy * 2,
                linewidth=1.6, edgecolor=col, facecolor=col,
                alpha=0.18 * growth, linestyle="--", zorder=2
            )
            ax.add_patch(rect)
    elif 15 <= f <= 22:
        # Pruning stage: prune boxes flash red & fade
        flash_prog = (f - 15) / 7.0
        for b in init_boxes:
            v1, w1, v2, w2 = b["box"]
            col = CLASS_COLORS[b["c"]]
            if b["status"] == "prune":
                # Dissolving
                fade = max(0.0, 1.0 - flash_prog)
                rect = patches.Rectangle(
                    (v1, w1), v2 - v1, w2 - w1,
                    linewidth=2.0 * fade, edgecolor="#ff3366", facecolor="#ff3366",
                    alpha=0.35 * fade, linestyle=":", zorder=3
                )
                ax.add_patch(rect)
                if fade > 0.3:
                    ax.text((v1 + v2) / 2, (w1 + w2) / 2, "PRUNED", color="#ff5577",
                            fontsize=7, fontweight="bold", ha="center", va="center", zorder=4)
            else:
                rect = patches.Rectangle(
                    (v1, w1), v2 - v1, w2 - w1,
                    linewidth=1.6, edgecolor=col, facecolor=col,
                    alpha=0.18, linestyle="--", zorder=2
                )
                ax.add_patch(rect)
    elif 23 <= f <= 29:
        # Merging stage: merge boxes transition into merged_boxes
        m_prog = (f - 23) / 6.0
        for b in merged_boxes:
            v1, w1, v2, w2 = b["box"]
            col = CLASS_COLORS[b["c"]]
            rect = patches.Rectangle(
                (v1, w1), v2 - v1, w2 - w1,
                linewidth=2.2, edgecolor=col, facecolor=col,
                alpha=0.22, linestyle="-", zorder=2
            )
            ax.add_patch(rect)
            # Glow border
            glow = patches.Rectangle(
                (v1 - 0.01, w1 - 0.01), (v2 - v1) + 0.02, (w2 - w1) + 0.02,
                linewidth=0.8, edgecolor=col, facecolor="none",
                alpha=0.4 * m_prog, linestyle="-", zorder=2
            )
            ax.add_patch(glow)
    elif f >= 30:
        # Hybrid decision stage: refined boxes with fuzzy contour glow
        for b in merged_boxes:
            v1, w1, v2, w2 = b["box"]
            col = CLASS_COLORS[b["c"]]
            # Outer membership contour (gamma decay)
            for ring, r_alpha in [(0.04, 0.08), (0.02, 0.14)]:
                contour = patches.Rectangle(
                    (v1 - ring, w1 - ring), (v2 - v1) + ring * 2, (w2 - w1) + ring * 2,
                    linewidth=0.8, edgecolor=col, facecolor=col,
                    alpha=r_alpha, zorder=2
                )
                ax.add_patch(contour)
            rect = patches.Rectangle(
                (v1, w1), v2 - v1, w2 - w1,
                linewidth=2.5, edgecolor=col, facecolor=col,
                alpha=0.25, linestyle="-", zorder=3
            )
            ax.add_patch(rect)
            # Label box
            ax.text(v1 + 0.015, w2 - 0.035, f"B_{b['c']}: Class {b['c']}\n[Pure: 100%]",
                    color=TEXT_COLOR, fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_COLOR, edgecolor=col, alpha=0.85),
                    zorder=5)

    # Plot Nodes
    for c in range(len(centers)):
        idx_train = np.where((labels == c) & train_mask)[0]
        idx_test = np.where((labels == c) & test_mask)[0]
        col = CLASS_COLORS[c]

        # Training nodes (squares with solid fill)
        ax.scatter(nodes[idx_train, 0], nodes[idx_train, 1],
                   s=75, color=col, edgecolors="#ffffff", linewidth=1.4,
                   marker="s", label=f"{CLASS_NAMES[c]} (Train)" if f == 0 else "",
                   zorder=6)

        # Test nodes (circles)
        ax.scatter(nodes[idx_test, 0], nodes[idx_test, 1],
                   s=45, color=col, alpha=0.85, edgecolors=BG_COLOR, linewidth=0.8,
                   marker="o", label=f"{CLASS_NAMES[c]} (Test)" if f == 0 else "",
                   zorder=5)

    # Header Card overlay
    ax.text(0.02, 0.965, phase_title, color=TEXT_COLOR, fontsize=12.5, fontweight="bold", va="top")
    ax.text(0.02, 0.915, phase_sub, color=TEXT_MUTED, fontsize=8.5, va="top")

    # Phase Badge on top right
    ax.text(0.98, 0.965, f" {phase_badge} ", color="#ffffff", fontsize=8, fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=badge_color, edgecolor="none", alpha=0.9))

    # Metric HUD on bottom left
    if f <= 6:
        hud_text = "GNN CE Loss: 0.248 | Edge Homophily: 81.0% | Embedding Dim: 64-D"
    elif f <= 14:
        hud_text = "FMNN Boxes: 12 | Mean Vol: 0.018 | Cross-Class Overlaps: 0 | Acc: 77.6%"
    elif f <= 22:
        hud_text = "Pruning Threshold: tau_p=0.01 | Dropped: 4 redundant boxes | Acc: 79.6%"
    elif f <= 29:
        hud_text = "Final Boxes: 4 | Compactness: +66.7% | Zero Class Overlaps (Simpson Safe)"
    else:
        hud_text = "Final Model: B14_final | Test Acc: 81.8% | Boxes: 7.8 | Decision Purity: 98.2%"

    ax.text(0.02, 0.025, hud_text, color=TEXT_COLOR, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1e293b", edgecolor=GRID_COLOR, alpha=0.9),
            va="bottom", zorder=7)

    # Bottom Right Legend
    ax.text(0.98, 0.025, "[s] Train Node   [o] Test Node   -- Hyperbox [V, W]",
            color=TEXT_MUTED, fontsize=7.5, ha="right", va="bottom", zorder=7)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Save frame to buffer
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frames.append(Image.fromarray(rgba))
    plt.close(fig)

# Save animated GIF (duration=280ms per frame = smooth ~10 sec loop)
frames[0].save(
    GIF_PATH,
    save_all=True,
    append_images=frames[1:],
    duration=280,
    loop=0,
    optimize=True
)

print(f"Generated animated GIF at {GIF_PATH} ({GIF_PATH.stat().st_size / 1024:.1f} KB)")
