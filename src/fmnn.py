"""Fuzzy Min-Max Neural Network (FMNN) implemented from scratch in NumPy."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HyperBox:
    """Hyperbox representation defined by min point v and max point w."""

    id: int
    v: np.ndarray
    w: np.ndarray
    cls: int
    members: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.v = np.asarray(self.v, dtype=np.float64)
        self.w = np.asarray(self.w, dtype=np.float64)


class FMNN:
    """Fuzzy Min-Max Neural Network classifier (Simpson 1992)."""

    def __init__(self, theta: float = 0.3, gamma: float = 4.0) -> None:
        self.theta = float(theta)
        self.gamma = float(gamma)
        self.boxes: dict[int, HyperBox] = {}
        self.next_id: int = 1
        self.num_classes: int = 0
        self.n_dims: int = 0

    def membership(self, a: np.ndarray, box: HyperBox) -> float:
        """Compute membership of a single point a in a hyperbox (Simpson eq 1).

        Args:
            a: 1D array of shape (n_dims,).
            box: HyperBox instance.

        Returns:
            Membership value in [0, 1]. Exactly 1.0 if a is inside the box.
        """
        a = np.asarray(a, dtype=np.float64)
        n = len(a)
        diff_w = np.clip(a - box.w, 0.0, 1.0)
        penalty_w = np.clip(1.0 - np.clip(self.gamma * diff_w, 0.0, None), 0.0, None)

        diff_v = np.clip(box.v - a, 0.0, 1.0)
        penalty_v = np.clip(1.0 - np.clip(self.gamma * diff_v, 0.0, None), 0.0, None)

        return float(np.sum(penalty_w + penalty_v) / (2.0 * n))

    def membership_matrix(
        self,
        Z: np.ndarray,
        box_list: list[HyperBox] | None = None,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Compute membership matrix in memory-bounded chunks using NumPy broadcasting.

        Args:
            Z: Array of shape (num_points, n_dims).
            box_list: List of HyperBox instances. If None, uses sorted self.boxes.values().
            chunk_size: Maximum points per vectorized chunk to bound peak memory.

        Returns:
            Array of shape (num_points, num_boxes) with values in [0, 1].
        """
        Z = np.asarray(Z, dtype=np.float64)
        if box_list is None:
            box_list = [self.boxes[k] for k in sorted(self.boxes.keys())]

        num_points, n = Z.shape
        num_boxes = len(box_list)
        if num_boxes == 0:
            return np.zeros((num_points, 0), dtype=np.float64)

        V = np.stack([b.v for b in box_list], axis=0)  # (num_boxes, n)
        W = np.stack([b.w for b in box_list], axis=0)  # (num_boxes, n)
        out = np.empty((num_points, num_boxes), dtype=np.float64)

        step = max(1, int(chunk_size))
        for start in range(0, num_points, step):
            end = min(start + step, num_points)
            z_chunk = Z[start:end, None, :]
            diff_w = np.clip(z_chunk - W[None, :, :], 0.0, 1.0)
            penalty_w = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * diff_w))

            diff_v = np.clip(V[None, :, :] - z_chunk, 0.0, 1.0)
            penalty_v = np.maximum(0.0, 1.0 - np.maximum(0.0, self.gamma * diff_v))

            mem_chunk = np.sum(penalty_w + penalty_v, axis=-1) / (2.0 * n)
            out[start:end] = np.clip(mem_chunk, 0.0, 1.0)
            del z_chunk, diff_w, penalty_w, diff_v, penalty_v, mem_chunk

        del V, W
        return out

    def class_scores(self, Z: np.ndarray, boxes: list[HyperBox] | None = None) -> np.ndarray:
        """Compute fuzzy class scores for all points (Simpson eq 2).

        A class with no hyperboxes receives a score of 0.0.

        Args:
            Z: Array of shape (num_points, n_dims).
            boxes: Optional subset of boxes. If None, uses all boxes in self.boxes.

        Returns:
            Array of shape (num_points, num_classes).
        """
        Z = np.asarray(Z, dtype=np.float64)
        num_points = Z.shape[0]
        scores = np.zeros((num_points, self.num_classes), dtype=np.float64)

        box_pool = list(self.boxes.values()) if boxes is None else boxes
        if not box_pool:
            return scores

        # Evaluate all candidate boxes in a single pass of membership_matrix
        mat = self.membership_matrix(Z, box_pool)
        for c in range(self.num_classes):
            c_indices = [i for i, b in enumerate(box_pool) if b.cls == c]
            if c_indices:
                scores[:, c] = np.max(mat[:, c_indices], axis=1)

        return scores

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Predict class labels for points, breaking ties deterministically by lowest class index."""
        scores = self.class_scores(Z)
        return np.argmax(scores, axis=1)

    def predict_with_confidence(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict class labels with confidence score and tie indicator.

        Returns:
            predictions: Predicted class indices.
            best_scores: Winning class membership score.
            is_tie: Boolean array indicating whether top two class scores tie or are within 1e-12.
        """
        scores = self.class_scores(Z)
        predictions = np.argmax(scores, axis=1)
        best_scores = np.max(scores, axis=1)

        if self.num_classes <= 1:
            is_tie = np.zeros(len(Z), dtype=bool)
        else:
            sorted_scores = np.partition(scores, -2, axis=1)
            second_best = sorted_scores[:, -2]
            is_tie = np.isclose(best_scores, second_best, atol=1e-12)

        return predictions, best_scores, is_tie

    def _test_and_contract(self, box_j: HyperBox, box_k: HyperBox) -> bool:
        """Test for overlap between expanded/new box j and other box k, and contract if needed.

        Returns:
            True if contraction occurred, False otherwise.
        """
        n = self.n_dims
        delta_old = 1.0
        D = -1

        for i in range(n):
            v_j, w_j = box_j.v[i], box_j.w[i]
            v_k, w_k = box_k.v[i], box_k.w[i]
            case = None
            delta_new = delta_old

            if v_j < v_k < w_j < w_k:
                case = 1
                delta_new = min(w_j - v_k, delta_old)
            elif v_k < v_j < w_k < w_j:
                case = 2
                delta_new = min(w_k - v_j, delta_old)
            elif v_j <= v_k <= w_k <= w_j and (v_j < v_k or w_k < w_j):
                case = 3
                delta_new = min(min(w_k - v_j, w_j - v_k), delta_old)
            elif v_k <= v_j <= w_j <= w_k and (v_k < v_j or w_j < w_k):
                case = 4
                delta_new = min(min(w_j - v_k, w_k - v_j), delta_old)

            if delta_old - delta_new > 0.0:
                D = i
                delta_old = delta_new
            elif case is None:
                D = -1
                break

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

            # Bound coordinates to [0, 1] and keep v <= w
            box_j.v = np.clip(box_j.v, 0.0, 1.0)
            box_j.w = np.clip(box_j.w, 0.0, 1.0)
            box_k.v = np.clip(box_k.v, 0.0, 1.0)
            box_k.w = np.clip(box_k.w, 0.0, 1.0)
            box_j.v = np.minimum(box_j.v, box_j.w)
            box_k.v = np.minimum(box_k.v, box_k.w)
            return True

        return False

    def fit(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        node_ids: list[int] | np.ndarray,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> FMNN:
        """Fit base FMNN on training points using single-pass expansion and contraction.

        Args:
            Z_train: Array of shape (M, n_dims) with normalized features in [0, 1].
            y_train: Array of shape (M,) with class labels.
            node_ids: Array or list of length M with original graph node ids.
            shuffle: If True, shuffles sample processing order using seed.
            seed: Random seed for optional shuffle.
        """
        Z_train = np.asarray(Z_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.int64)
        node_ids = list(node_ids)

        M, n = Z_train.shape
        self.n_dims = n
        self.num_classes = int(np.max(y_train)) + 1
        self.boxes.clear()
        self.next_id = 1

        indices = np.arange(M)
        if shuffle:
            rng = np.random.RandomState(seed if seed is not None else 42)
            rng.shuffle(indices)

        for idx in indices:
            a = Z_train[idx]
            d = int(y_train[idx])
            node_id = int(node_ids[idx])

            c_boxes = self.boxes_of_class(d)
            if not c_boxes:
                # Step 3a: Create new box
                box = HyperBox(
                    id=self.next_id,
                    v=a.copy(),
                    w=a.copy(),
                    cls=d,
                    members=[node_id],
                )
                self.boxes[self.next_id] = box
                changed_box = box
                self.next_id += 1
            else:
                # Step 3b: Vectorized membership ranking and expansion check
                if len(c_boxes) > 1:
                    mems = self.membership_matrix(a[None, :], c_boxes)[0]
                    order = sorted(range(len(c_boxes)), key=lambda i: (-mems[i], c_boxes[i].id))
                    ordered_boxes = [c_boxes[i] for i in order]
                else:
                    ordered_boxes = c_boxes

                expanded_box: HyperBox | None = None
                for b in ordered_boxes:
                    new_v = np.minimum(b.v, a)
                    new_w = np.maximum(b.w, a)
                    if n * self.theta >= np.sum(new_w - new_v):
                        expanded_box = b
                        b.v = new_v
                        b.w = new_w
                        b.members.append(node_id)
                        break

                if expanded_box is None:
                    box = HyperBox(
                        id=self.next_id,
                        v=a.copy(),
                        w=a.copy(),
                        cls=d,
                        members=[node_id],
                    )
                    self.boxes[self.next_id] = box
                    changed_box = box
                    self.next_id += 1
                else:
                    changed_box = expanded_box

            # Step 3d: Vectorized candidate prefilter before contraction against other-class boxes
            diff_boxes = [b for b in self.boxes.values() if b.cls != changed_box.cls]
            if diff_boxes:
                diff_boxes.sort(key=lambda b: b.id)
                V_d = np.stack([b.v for b in diff_boxes], axis=0)
                W_d = np.stack([b.w for b in diff_boxes], axis=0)
                cand_mask = np.all((changed_box.v <= W_d) & (V_d <= changed_box.w), axis=1)
                for idx_k in np.nonzero(cand_mask)[0]:
                    self._test_and_contract(changed_box, diff_boxes[int(idx_k)])
                del V_d, W_d, cand_mask

        # Step 4: Member reassignment after all contractions
        self.reassign_members(Z_train, y_train, node_ids)
        return self

    # Fit labeled training nodes first, then absorb pseudo-labeled nodes without creating redundant boxes.
    def fit_with_pseudo(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        train_ids: list[int] | np.ndarray,
        Z_pseudo: np.ndarray,
        y_pseudo: np.ndarray,
        pseudo_ids: list[int] | np.ndarray,
        allow_new_boxes: bool = True,
    ) -> FMNN:
        """Fit on true train nodes first, then add pseudo nodes preferring full-coverage or expansion."""
        self.fit(Z_train, y_train, train_ids)

        Z_tr = np.asarray(Z_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.int64)
        ids_tr = [int(i) for i in train_ids]

        Z_p = np.asarray(Z_pseudo, dtype=np.float64)
        y_p = np.asarray(y_pseudo, dtype=np.int64)
        ids_p = [int(i) for i in pseudo_ids]

        if len(Z_p) == 0:
            return self

        n = self.n_dims
        for idx in range(len(Z_p)):
            a = Z_p[idx]
            d = int(y_p[idx])
            nid = ids_p[idx]
            c_boxes = self.boxes_of_class(d)
            if not c_boxes:
                continue  # Never introduce a class absent from training boxes

            V_c = np.stack([b.v for b in c_boxes], axis=0)
            W_c = np.stack([b.w for b in c_boxes], axis=0)

            # 1) If an existing same-class box already fully covers a, attach without new box creation.
            covered_mask = np.all((V_c - 1e-12 <= a[None, :]) & (a[None, :] <= W_c + 1e-12), axis=1)
            if np.any(covered_mask):
                cov_idx = int(np.nonzero(covered_mask)[0][0])
                c_boxes[cov_idx].members.append(nid)
                del V_c, W_c, covered_mask
                continue

            # 2) Try expanding highest-membership same-class box within theta.
            mems = self.membership_matrix(a[None, :], c_boxes)[0]
            order = sorted(range(len(c_boxes)), key=lambda i: (-mems[i], c_boxes[i].id))
            changed_box: HyperBox | None = None
            for i in order:
                b = c_boxes[i]
                new_v = np.minimum(b.v, a)
                new_w = np.maximum(b.w, a)
                if n * self.theta >= np.sum(new_w - new_v):
                    b.v = new_v
                    b.w = new_w
                    b.members.append(nid)
                    changed_box = b
                    break

            # 3) Only if no same-class box covers or expands within theta, optionally create a new box.
            if changed_box is None and allow_new_boxes:
                new_box = HyperBox(
                    id=self.next_id,
                    v=a.copy(),
                    w=a.copy(),
                    cls=d,
                    members=[nid],
                )
                self.boxes[self.next_id] = new_box
                changed_box = new_box
                self.next_id += 1

            del V_c, W_c, covered_mask, mems
            if changed_box is not None:
                diff_boxes = [b for b in self.boxes.values() if b.cls != changed_box.cls]
                if diff_boxes:
                    diff_boxes.sort(key=lambda b: b.id)
                    V_d = np.stack([b.v for b in diff_boxes], axis=0)
                    W_d = np.stack([b.w for b in diff_boxes], axis=0)
                    cand_mask = np.all((changed_box.v <= W_d) & (V_d <= changed_box.w), axis=1)
                    for idx_k in np.nonzero(cand_mask)[0]:
                        self._test_and_contract(changed_box, diff_boxes[int(idx_k)])
                    del V_d, W_d, cand_mask

        Z_all = np.vstack([Z_tr, Z_p])
        y_all = np.concatenate([y_tr, y_p])
        ids_all = ids_tr + ids_p
        self.reassign_members(Z_all, y_all, ids_all)
        return self

    def reassign_members(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        node_ids: list[int] | np.ndarray,
        cached_train_mem: np.ndarray | None = None,
        id_to_col: dict[int, int] | None = None,
    ) -> dict[int, int]:
        """Reassign every training node to the box of its own class with highest membership (vectorized)."""
        for box in self.boxes.values():
            box.members.clear()

        Z_arr = np.asarray(Z_train, dtype=np.float64) if cached_train_mem is None else None
        y_arr = np.asarray(y_train, dtype=np.int64)
        ids_arr = np.asarray(node_ids, dtype=np.int64)
        assignments: dict[int, int] = {}

        for c in range(self.num_classes):
            c_boxes = self.boxes_of_class(c)
            if not c_boxes:
                continue
            c_rows = np.nonzero(y_arr == c)[0]
            if len(c_rows) == 0:
                continue

            if cached_train_mem is not None and id_to_col is not None:
                c_cols = [id_to_col[b.id] for b in c_boxes]
                if len(c_cols) == 1:
                    best_cols = np.zeros(len(c_rows), dtype=np.int64)
                else:
                    mat = cached_train_mem[np.ix_(c_rows, c_cols)]
                    best_cols = np.argmax(mat, axis=1)
            else:
                mat = self.membership_matrix(Z_arr[c_rows], c_boxes)
                best_cols = np.argmax(mat, axis=1)

            for row_pos, col_idx in zip(c_rows, best_cols):
                nid = int(ids_arr[int(row_pos)])
                box = c_boxes[int(col_idx)]
                box.members.append(nid)
                assignments[nid] = box.id

        return assignments

    def get_assignments(self) -> dict[int, int]:
        """Return dictionary mapping training node_id -> assigned box_id."""
        assignments: dict[int, int] = {}
        for box in self.boxes.values():
            for nid in box.members:
                assignments[nid] = box.id
        return assignments

    def num_boxes(self) -> int:
        """Return total number of hyperboxes."""
        return len(self.boxes)

    def boxes_per_class(self) -> dict[int, int]:
        """Return dictionary mapping class id -> number of hyperboxes."""
        counts = {c: 0 for c in range(self.num_classes)}
        for b in self.boxes.values():
            counts[b.cls] = counts.get(b.cls, 0) + 1
        return counts

    def boxes_of_class(self, c: int) -> list[HyperBox]:
        """Return list of hyperboxes of class c sorted by id."""
        return sorted([b for b in self.boxes.values() if b.cls == c], key=lambda b: b.id)

    def copy(self) -> FMNN:
        """Create a deep copy of the FMNN model with copied hyperboxes."""
        new_model = FMNN(theta=self.theta, gamma=self.gamma)
        new_model.num_classes = self.num_classes
        new_model.n_dims = self.n_dims
        new_model.next_id = self.next_id
        for bid, b in self.boxes.items():
            new_model.boxes[bid] = HyperBox(
                id=b.id,
                v=b.v.copy(),
                w=b.w.copy(),
                cls=b.cls,
                members=list(b.members),
            )
        return new_model

    def remove_box(self, box_id: int) -> None:
        """Remove a hyperbox by id."""
        self.boxes.pop(box_id, None)

    def add_box(self, box: HyperBox) -> None:
        """Add a hyperbox to the model and update book-keeping."""
        self.boxes[box.id] = box
        if box.id >= self.next_id:
            self.next_id = box.id + 1
        if box.cls >= self.num_classes:
            self.num_classes = box.cls + 1

    @staticmethod
    def box_volume_proxy(box: HyperBox) -> float:
        """Sum of side lengths proxy for hyperbox volume (avoids vanishing volume in high dimensions)."""
        return float(np.sum(box.w - box.v))

    @staticmethod
    def center(box: HyperBox) -> np.ndarray:
        """Center coordinate of the hyperbox."""
        return (box.v + box.w) / 2.0

    @staticmethod
    def boxes_overlap(box_a: HyperBox, box_b: HyperBox) -> bool:
        """Check if two hyperboxes strictly overlap in every dimension."""
        return bool(np.all(np.maximum(box_a.v, box_b.v) < np.minimum(box_a.w, box_b.w)))

    def cross_class_overlap_count(self) -> int:
        """Count pairs of hyperboxes of different classes that strictly overlap."""
        box_list = sorted(self.boxes.values(), key=lambda b: b.id)
        count = 0
        for i in range(len(box_list)):
            for j in range(i + 1, len(box_list)):
                b1, b2 = box_list[i], box_list[j]
                if b1.cls != b2.cls and self.boxes_overlap(b1, b2):
                    count += 1
        return count
