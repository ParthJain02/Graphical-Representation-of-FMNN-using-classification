"""
========================================================================================
Fuzzy Min-Max Neural Network (FMNN) Classifier - Standalone Practice Code
Implementation based on Patrick K. Simpson (1992):
"Fuzzy Min-Max Neural Networks - Part 1: Classification", IEEE Trans. Neural Networks.
========================================================================================

This script is completely self-contained. It requires only standard scientific packages:
  - numpy
  - matplotlib
  - scikit-learn (used only for standard dataset loading / splitting)

Core Concepts of FMNN:
1. Hyperbox Representation:
   Each hyperbox B_j is defined by its min corner V_j and max corner W_j in [0, 1]^n.
2. Membership Function (Simpson Eq. 1):
   Measures the fuzzy degree (0 to 1) to which an input pattern x belongs to hyperbox B_j.
   If x is inside B_j, membership is exactly 1.0. If outside, it decays at rate gamma.
3. Three-Step Learning:
   - Expansion: Find existing same-class box that can expand to include x without exceeding theta.
     If none found, create a new hyperbox initialized at x.
   - Overlap Test: Simpson's 4 geometric overlap cases to test if expanded box overlaps
     with another box of a DIFFERENT class.
   - Contraction: Contract along the single dimension of minimum overlap using minimal disturbance.
4. Classification:
   Winner-take-all based on the maximum fuzzy membership score across all hyperboxes.

Usage:
  python standalone_fmnn.py
========================================================================================
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass
class HyperBox:
    """Represents a single n-dimensional hyperbox in the FMNN.

    Attributes:
        id: Unique identifier for the hyperbox.
        v: Min coordinate vector (lower-left corner), shape (n_dims,).
        w: Max coordinate vector (upper-right corner), shape (n_dims,).
        cls: Class label integer that this hyperbox belongs to.
        members: List of training sample indices assigned to this hyperbox.
    """
    id: int
    v: np.ndarray
    w: np.ndarray
    cls: int
    members: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.v = np.asarray(self.v, dtype=np.float64)
        self.w = np.asarray(self.w, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        """Center coordinate of the hyperbox."""
        return (self.v + self.w) / 2.0

    @property
    def volume(self) -> float:
        """Geometric volume of the hyperbox."""
        return float(np.prod(np.maximum(0.0, self.w - self.v)))

    @property
    def size(self) -> float:
        """Sum of side lengths (L1 size)."""
        return float(np.sum(np.maximum(0.0, self.w - self.v)))


class StandaloneFMNN:
    """Fuzzy Min-Max Neural Network Classifier (Simpson 1992).

    Parameters:
        theta: Maximum hyperbox expansion size threshold (0.0 < theta <= 1.0).
               Smaller theta produces more, tighter hyperboxes (less general, higher resolution).
               Larger theta produces fewer, larger hyperboxes (more general).
        gamma: Sensitivity parameter regulating how fast fuzzy membership decays outside
               the hyperbox (gamma > 0, typically 1.0 to 8.0, default 4.0).
    """

    def __init__(self, theta: float = 0.3, gamma: float = 4.0) -> None:
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.theta: float = float(theta)
        self.gamma: float = float(gamma)
        self.boxes: dict[int, HyperBox] = {}
        self.next_id: int = 1
        self.num_classes: int = 0
        self.n_dims: int = 0

    def membership(self, a: np.ndarray, box: HyperBox) -> float:
        """Compute fuzzy membership b_j(a) of point a in box j (Simpson 1992 Eq. 1).

        b_j(a) = (1 / 2n) * sum_i [ max(0, 1 - max(0, gamma * min(1, a_i - w_ji)))
                                  + max(0, 1 - max(0, gamma * min(1, v_ji - a_i))) ]

        Returns:
            Membership value in [0.0, 1.0]. Exactly 1.0 if a is inside the hyperbox.
        """
        a = np.asarray(a, dtype=np.float64)
        n = len(a)

        # Upper boundary penalty: active when a_i > w_i
        diff_w = np.clip(a - box.w, 0.0, 1.0)
        penalty_w = np.clip(1.0 - np.clip(self.gamma * diff_w, 0.0, None), 0.0, None)

        # Lower boundary penalty: active when a_i < v_i
        diff_v = np.clip(box.v - a, 0.0, 1.0)
        penalty_v = np.clip(1.0 - np.clip(self.gamma * diff_v, 0.0, None), 0.0, None)

        return float(np.sum(penalty_w + penalty_v) / (2.0 * n))

    def membership_matrix(self, Z: np.ndarray, box_list: list[HyperBox] | None = None) -> np.ndarray:
        """Compute membership of multiple points against multiple boxes using vectorization.

        Args:
            Z: Array of shape (N, n_dims) where each row is an input vector in [0, 1]^n.
            box_list: List of HyperBox objects. If None, uses all active boxes.

        Returns:
            Array of shape (N, num_boxes) with membership values in [0.0, 1.0].
        """
        Z = np.asarray(Z, dtype=np.float64)
        if box_list is None:
            box_list = [self.boxes[k] for k in sorted(self.boxes.keys())]

        num_points, n = Z.shape
        num_boxes = len(box_list)
        if num_boxes == 0:
            return np.zeros((num_points, 0), dtype=np.float64)

        V = np.stack([b.v for b in box_list], axis=0)  # (B, n)
        W = np.stack([b.w for b in box_list], axis=0)  # (B, n)

        # Vectorized broadcasting: (N, 1, n) - (1, B, n) -> (N, B, n)
        diff_w = np.clip(Z[:, None, :] - W[None, :, :], 0.0, 1.0)
        penalty_w = np.clip(1.0 - np.clip(self.gamma * diff_w, 0.0, None), 0.0, None)

        diff_v = np.clip(V[None, :, :] - Z[:, None, :], 0.0, 1.0)
        penalty_v = np.clip(1.0 - np.clip(self.gamma * diff_v, 0.0, None), 0.0, None)

        membership = np.sum(penalty_w + penalty_v, axis=-1) / (2.0 * n)
        return np.clip(membership, 0.0, 1.0)

    def class_scores(self, Z: np.ndarray) -> np.ndarray:
        """Compute fuzzy class scores for all points (Simpson Eq. 2).

        score(c | x) = max_{j in Class c} b_j(x)

        Returns:
            Array of shape (N, num_classes) where entry [i, c] is class c's membership score for x_i.
        """
        Z = np.asarray(Z, dtype=np.float64)
        num_points = Z.shape[0]
        scores = np.zeros((num_points, self.num_classes), dtype=np.float64)

        for c in range(self.num_classes):
            c_boxes = [b for b in self.boxes.values() if b.cls == c]
            if c_boxes:
                mat = self.membership_matrix(Z, c_boxes)
                scores[:, c] = np.max(mat, axis=1)

        return scores

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict class labels using winner-take-all rule."""
        scores = self.class_scores(Z)
        return np.argmax(scores, axis=1)

    def _test_and_contract(self, box_j: HyperBox, box_k: HyperBox) -> bool:
        """Simpson (1992) Overlap Test and Minimal Disturbance Contraction.

        Tests whether expanded box j overlaps with different-class box k.
        If overlap exists across all dimensions, contracts along the dimension
        with minimum overlap (minimal disturbance).
        """
        n = self.n_dims
        delta_old = 1.0
        D = -1

        for i in range(n):
            v_j, w_j = box_j.v[i], box_j.w[i]
            v_k, w_k = box_k.v[i], box_k.w[i]
            case = None
            delta_new = delta_old

            # Case 1: v_j < v_k < w_j < w_k
            if v_j < v_k < w_j < w_k:
                case = 1
                delta_new = min(w_j - v_k, delta_old)
            # Case 2: v_k < v_j < w_k < w_j
            elif v_k < v_j < w_k < w_j:
                case = 2
                delta_new = min(w_k - v_j, delta_old)
            # Case 3: v_j <= v_k <= w_k <= w_j (box k contained inside box j)
            elif v_j <= v_k <= w_k <= w_j and (v_j < v_k or w_k < w_j):
                case = 3
                delta_new = min(min(w_k - v_j, w_j - v_k), delta_old)
            # Case 4: v_k <= v_j <= w_j <= w_k (box j contained inside box k)
            elif v_k <= v_j <= w_j <= w_k and (v_k < v_j or w_j < w_k):
                case = 4
                delta_new = min(min(w_j - v_k, w_k - v_j), delta_old)

            if delta_old - delta_new > 0.0:
                D = i
                delta_old = delta_new
            elif case is None:
                # If there is no overlap in at least one dimension, hyperboxes do not intersect
                D = -1
                break

        # Contraction along dimension D of minimum overlap
        if D != -1:
            v_j, w_j = box_j.v[D], box_j.w[D]
            v_k, w_k = box_k.v[D], box_k.w[D]

            if v_j < v_k < w_j < w_k:
                mid = (w_j + v_k) / 2.0
                box_j.w[D] = mid
                box_k.v[D] = mid
            elif v_k < v_j < w_k < w_j:
                mid = (w_k + v_j) / 2.0
                box_k.w[D] = mid
                box_j.v[D] = mid
            elif v_j <= v_k <= w_k <= w_j:
                if (w_k - v_j) < (w_j - v_k):
                    box_j.v[D] = w_k
                else:
                    box_j.w[D] = v_k
            elif v_k <= v_j <= w_j <= w_k:
                if (w_k - v_j) < (w_j - v_k):
                    box_k.w[D] = v_j
                else:
                    box_k.v[D] = w_j

            # Clip within [0, 1] and ensure v <= w
            box_j.v = np.clip(box_j.v, 0.0, 1.0)
            box_j.w = np.clip(box_j.w, 0.0, 1.0)
            box_k.v = np.clip(box_k.v, 0.0, 1.0)
            box_k.w = np.clip(box_k.w, 0.0, 1.0)
            box_j.v = np.minimum(box_j.v, box_j.w)
            box_k.v = np.minimum(box_k.v, box_k.w)
            return True

        return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> StandaloneFMNN:
        """Train FMNN on normalized data X in [0, 1]^n and class labels y.

        Args:
            X: Training samples of shape (N, n_dims), normalized to [0, 1].
            y: Integer class labels of shape (N,).

        Returns:
            Self (fitted model).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if np.any(X < 0.0) or np.any(X > 1.0):
            raise ValueError("All features must be normalized to [0, 1] before fitting FMNN.")

        num_points, n = X.shape
        self.n_dims = n
        self.num_classes = int(np.max(y)) + 1
        self.boxes.clear()
        self.next_id = 1

        for idx in range(num_points):
            x = X[idx]
            c = int(y[idx])

            # Step 1: Find candidate same-class boxes sorted by highest membership
            same_class_boxes = [b for b in self.boxes.values() if b.cls == c]
            expanded = False

            if same_class_boxes:
                # Calculate memberships to sort candidates
                m_scores = [self.membership(x, b) for b in same_class_boxes]
                sorted_indices = np.argsort(m_scores)[::-1]

                for s_idx in sorted_indices:
                    box = same_class_boxes[s_idx]
                    # Test expansion criterion: each coordinate expansion must be <= theta
                    v_cand = np.minimum(box.v, x)
                    w_cand = np.maximum(box.w, x)

                    if np.all((w_cand - v_cand) <= self.theta + 1e-12):
                        # Expand hyperbox
                        box.v = v_cand
                        box.w = w_cand
                        box.members.append(idx)
                        expanded = True

                        # Step 2 & 3: Test and contract against all other-class boxes
                        for other_box in list(self.boxes.values()):
                            if other_box.cls != c and other_box.id != box.id:
                                self._test_and_contract(box, other_box)
                        break

            # If no suitable box could expand without violating theta, create a new hyperbox
            if not expanded:
                new_box = HyperBox(
                    id=self.next_id,
                    v=x.copy(),
                    w=x.copy(),
                    cls=c,
                    members=[idx],
                )
                self.boxes[self.next_id] = new_box
                self.next_id += 1

                # Overlap test and contraction for the newly created box
                for other_box in list(self.boxes.values()):
                    if other_box.cls != c and other_box.id != new_box.id:
                        self._test_and_contract(new_box, other_box)

        return self


# ======================================================================================
# Demonstration and Practice Script
# ======================================================================================

def run_fmnn_demo() -> None:
    """Run educational FMNN training on 2D synthetic clusters and Iris dataset."""
    print("=" * 75)
    print("FUZZY MIN-MAX NEURAL NETWORK (FMNN) - SIMPSON 1992 PRACTICE DEMO")
    print("=" * 75)

    # 1. Generate 2D Multi-Class Synthetic Classification Problem
    np.random.seed(42)
    print("\n[Step 1] Generating 2D synthetic dataset with 3 classes for clear visualization...")
    n_samples_per_class = 60
    
    # Class 0: bottom-left
    c0 = np.random.randn(n_samples_per_class, 2) * 0.4 + np.array([2.0, 2.0])
    # Class 1: top-right
    c1 = np.random.randn(n_samples_per_class, 2) * 0.45 + np.array([5.5, 5.0])
    # Class 2: center-right
    c2 = np.random.randn(n_samples_per_class, 2) * 0.35 + np.array([5.0, 2.0])

    X_raw = np.vstack([c0, c1, c2])
    y_raw = np.array([0] * n_samples_per_class + [1] * n_samples_per_class + [2] * n_samples_per_class)

    # Shuffle data
    perm = np.random.permutation(len(y_raw))
    X_raw, y_raw = X_raw[perm], y_raw[perm]

    # Split 70% Train, 30% Test
    split_idx = int(0.7 * len(X_raw))
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test = y_raw[:split_idx], y_raw[split_idx:]

    # Min-Max Scaling to [0, 1] strictly using Training statistics
    min_val = np.min(X_train_raw, axis=0)
    max_val = np.max(X_train_raw, axis=0)
    denom = np.where((max_val - min_val) < 1e-12, 1.0, max_val - min_val)

    X_train = np.clip((X_train_raw - min_val) / denom, 0.0, 1.0)
    X_test = np.clip((X_test_raw - min_val) / denom, 0.0, 1.0)

    print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"  Feature dimensions: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")

    # 2. Train FMNN Classifier
    theta = 0.25  # Max hyperbox size
    gamma = 4.0   # Sensitivity parameter
    print(f"\n[Step 2] Initializing FMNN with theta={theta}, gamma={gamma}...")
    fmnn = StandaloneFMNN(theta=theta, gamma=gamma)
    fmnn.fit(X_train, y_train)

    print(f"  Training complete! Total Hyperboxes Created: {len(fmnn.boxes)}")
    for b_id, box in fmnn.boxes.items():
        print(f"    - Box ID {b_id:2d}: Class {box.cls}, Members={len(box.members):2d}, "
              f"v=[{box.v[0]:.3f}, {box.v[1]:.3f}], w=[{box.w[0]:.3f}, {box.w[1]:.3f}], "
              f"side_lengths=[{(box.w[0]-box.v[0]):.3f}, {(box.w[1]-box.v[1]):.3f}]")

    # 3. Evaluate Predictions
    y_pred_train = fmnn.predict(X_train)
    y_pred_test = fmnn.predict(X_test)
    train_acc = np.mean(y_pred_train == y_train) * 100.0
    test_acc = np.mean(y_pred_test == y_test) * 100.0

    print(f"\n[Step 3] Performance Results:")
    print(f"  Training Accuracy: {train_acc:.2f}%")
    print(f"  Testing Accuracy:  {test_acc:.2f}%")

    # 4. Plot 2D Hyperbox Decision Boundaries
    print("\n[Step 4] Visualizing Learned Hyperboxes & Decision Space...")
    fig, ax = plt.subplots(figsize=(9, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    class_names = ['Class 0 (Blue)', 'Class 1 (Orange)', 'Class 2 (Green)']

    # Plot Background Fuzzy Decision Contours
    grid_res = 120
    gx, gy = np.meshgrid(np.linspace(0, 1, grid_res), np.linspace(0, 1, grid_res))
    grid_points = np.c_[gx.ravel(), gy.ravel()]
    grid_preds = fmnn.predict(grid_points).reshape(gx.shape)
    ax.contourf(gx, gy, grid_preds, levels=[-0.5, 0.5, 1.5, 2.5], colors=['#e6f2ff', '#fff3e6', '#eafaf1'], alpha=0.5)

    # Plot Training and Testing Points
    for c in range(3):
        # Train points (circles)
        mask_tr = (y_train == c)
        ax.scatter(X_train[mask_tr, 0], X_train[mask_tr, 1], c=colors[c], marker='o',
                   edgecolors='black', s=55, label=f'{class_names[c]} Train')
        # Test points (triangles)
        mask_te = (y_test == c)
        ax.scatter(X_test[mask_te, 0], X_test[mask_te, 1], c=colors[c], marker='^',
                   edgecolors='black', s=65, label=f'{class_names[c]} Test')

    # Draw Each Learned Hyperbox Rectangle
    for b_id, box in fmnn.boxes.items():
        width = box.w[0] - box.v[0]
        height = box.w[1] - box.v[1]
        rect = Rectangle(
            (box.v[0], box.v[1]), width, height,
            linewidth=2.2, edgecolor=colors[box.cls], facecolor='none',
            linestyle='--', label='_nolegend_'
        )
        ax.add_patch(rect)
        # Label box ID at center
        ax.text(box.center[0], box.center[1], f"B{box.id}",
                color=colors[box.cls], fontsize=10, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[box.cls], alpha=0.8))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalized Feature 1", fontsize=12)
    ax.set_ylabel("Normalized Feature 2", fontsize=12)
    ax.set_title(f"Fuzzy Min-Max Neural Network (Simpson 1992)\n"
                 f"$\\theta={theta}$, $\\gamma={gamma}$ | {len(fmnn.boxes)} Hyperboxes | Test Acc: {test_acc:.1f}%",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)

    output_fig_path = "fmnn_practice_decision_boundary.png"
    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=200)
    plt.close()
    print(f"  Saved 2D visualization plot to: {output_fig_path}")

    # 5. Optional Iris Benchmark
    try:
        from sklearn.datasets import load_iris
        print("\n[Step 5] Running FMNN on Classic Fisher Iris (4 Features, 3 Classes)...")
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        p_iris = np.random.RandomState(42).permutation(len(y_iris))
        X_iris, y_iris = X_iris[p_iris], y_iris[p_iris]

        tr_idx = int(0.7 * len(X_iris))
        X_tr_i, X_te_i = X_iris[:tr_idx], X_iris[tr_idx:]
        y_tr_i, y_te_i = y_iris[:tr_idx], y_iris[tr_idx:]

        # Normalize to [0, 1] using train set
        mn_i = np.min(X_tr_i, axis=0)
        mx_i = np.max(X_tr_i, axis=0)
        X_tr_i = np.clip((X_tr_i - mn_i) / (mx_i - mn_i), 0.0, 1.0)
        X_te_i = np.clip((X_te_i - mn_i) / (mx_i - mn_i), 0.0, 1.0)

        fmnn_iris = StandaloneFMNN(theta=0.25, gamma=4.0)
        fmnn_iris.fit(X_tr_i, y_tr_i)
        te_preds = fmnn_iris.predict(X_te_i)
        te_acc_iris = np.mean(te_preds == y_te_i) * 100.0

        print(f"  Iris FMNN Fit Successful: {len(fmnn_iris.boxes)} Hyperboxes learned.")
        print(f"  Iris Test Accuracy: {te_acc_iris:.2f}%")
    except Exception as e:
        print(f"  Note: Iris evaluation skipped ({e}).")

    print("\n" + "=" * 75)
    print("DEMO COMPLETE! You can adjust theta and gamma above to practice FMNN learning.")
    print("=" * 75)


if __name__ == "__main__":
    run_fmnn_demo()
