"""Graph statistics and caching utilities for FMNN hyperbox refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from src.fmnn import FMNN, HyperBox


def internal_edges(S: list[int] | np.ndarray, adj_csr: sp.csr_matrix) -> int:
    """Number of undirected graph edges with both endpoints in S (each edge counted once)."""
    if len(S) <= 1:
        return 0
    S_arr = np.asarray(S, dtype=np.int64)
    sub = adj_csr[S_arr, :][:, S_arr]
    return int(sub.nnz // 2)


def volume(S: list[int] | np.ndarray, degree: np.ndarray) -> int:
    """Sum of degrees of the nodes in set S."""
    if len(S) == 0:
        return 0
    S_arr = np.asarray(S, dtype=np.int64)
    return int(np.sum(degree[S_arr]))


def boundary_edges(
    S: list[int] | np.ndarray,
    adj_csr: sp.csr_matrix,
    degree: np.ndarray | None = None,
) -> int:
    """Number of undirected graph edges with exactly one endpoint in S."""
    if len(S) == 0:
        return 0
    S_arr = np.asarray(S, dtype=np.int64)
    if degree is not None:
        vol = int(np.sum(degree[S_arr]))
    else:
        vol = int(adj_csr[S_arr, :].sum())
    intern = internal_edges(S_arr, adj_csr)
    return int(vol - 2 * intern)


def conductance(
    S: list[int] | np.ndarray,
    adj_csr: sp.csr_matrix,
    degree: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Conductance of set S: boundary / (2 * internal + boundary + eps)."""
    if len(S) == 0:
        return 0.0
    S_arr = np.asarray(S, dtype=np.int64)
    intern = internal_edges(S_arr, adj_csr)
    bound = boundary_edges(S_arr, adj_csr, degree=degree)
    denom = 2.0 * intern + bound + eps
    if denom <= eps:
        return 0.0
    return float(bound / denom)


def class_purity(
    S: list[int] | np.ndarray,
    box_class: int,
    y: np.ndarray | Any,
    train_mask: np.ndarray | Any,
) -> float:
    """Fraction of training nodes in S whose ground-truth label equals box_class.

    WARNING: Only training nodes are read to prevent label leakage.
    """
    if len(S) == 0:
        return 1.0

    S_arr = np.asarray(S, dtype=np.int64)
    mask = np.asarray(train_mask)
    labels = np.asarray(y)

    train_members = S_arr[mask[S_arr]]
    if len(train_members) == 0:
        return 0.0

    matches = np.sum(labels[train_members] == box_class)
    return float(matches / len(train_members))


def neighbor_label_agreement(
    S: list[int] | np.ndarray,
    box_class: int,
    adj_csr: sp.csr_matrix,
    y: np.ndarray | Any,
    train_mask: np.ndarray | Any,
    exclude_self: bool = True,
) -> tuple[float, int]:
    """Fraction of labeled training neighbors of S whose ground-truth label equals box_class.

    Returns:
        (agreement, labeled_neighbors_count). If no labeled neighbors exist, returns (0.5, 0).
    """
    if len(S) == 0:
        return 0.5, 0

    S_arr = np.asarray(S, dtype=np.int64)
    mask = np.asarray(train_mask)
    labels = np.asarray(y)

    sub = adj_csr[S_arr, :]
    all_nbrs = np.unique(sub.indices)

    if exclude_self:
        nbrs = np.setdiff1d(all_nbrs, S_arr)
    else:
        nbrs = all_nbrs

    if len(nbrs) == 0:
        return 0.5, 0

    labeled_nbrs = nbrs[mask[nbrs]]
    total_labeled = len(labeled_nbrs)
    if total_labeled == 0:
        return 0.5, 0

    matches = np.sum(labels[labeled_nbrs] == box_class)
    agreement = float(matches / total_labeled)
    return agreement, total_labeled


def edges_between(
    S_i: list[int] | np.ndarray,
    S_j: list[int] | np.ndarray,
    adj_csr: sp.csr_matrix,
) -> int:
    """Number of undirected graph edges with one endpoint in S_i and the other in S_j."""
    if len(S_i) == 0 or len(S_j) == 0:
        return 0
    S_i_arr = np.asarray(S_i, dtype=np.int64)
    S_j_arr = np.asarray(S_j, dtype=np.int64)
    return int(adj_csr[S_i_arr, :][:, S_j_arr].nnz)


def graph_association(
    S_i: list[int] | np.ndarray,
    S_j: list[int] | np.ndarray,
    adj_csr: sp.csr_matrix,
    degree: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Normalized graph association between sets S_i and S_j in [0, 1]."""
    e_ij = edges_between(S_i, S_j, adj_csr)
    if e_ij == 0:
        return 0.0

    vol_i = volume(S_i, degree)
    vol_j = volume(S_j, degree)
    denom = np.sqrt(float(vol_i * vol_j) + eps)
    assoc = float(e_ij / denom)
    return float(np.clip(assoc, 0.0, 1.0))


def neighbor_jaccard(
    S_i: list[int] | np.ndarray,
    S_j: list[int] | np.ndarray,
    adj_csr: sp.csr_matrix,
) -> float:
    """Jaccard similarity of the neighbor sets N(S_i) and N(S_j)."""
    if len(S_i) == 0 or len(S_j) == 0:
        return 0.0
    S_i_arr = np.asarray(S_i, dtype=np.int64)
    S_j_arr = np.asarray(S_j, dtype=np.int64)

    nbrs_i = set(adj_csr[S_i_arr, :].indices)
    nbrs_j = set(adj_csr[S_j_arr, :].indices)

    union = len(nbrs_i | nbrs_j)
    if union == 0:
        return 0.0
    intersection = len(nbrs_i & nbrs_j)
    return float(intersection / union)


def build_box_adjacency(
    fmnn: FMNN,
    adj_csr: sp.csr_matrix,
) -> dict[tuple[int, int], int]:
    """Compute undirected edge counts between every connected pair of boxes (i < j).

    Computed efficiently via sparse matrix multiplication B^T @ adj_csr @ B.
    """
    box_ids = sorted(fmnn.boxes.keys())
    num_boxes = len(box_ids)
    if num_boxes <= 1:
        return {}

    num_nodes = adj_csr.shape[0]
    id_to_idx = {bid: idx for idx, bid in enumerate(box_ids)}

    row_indices: list[int] = []
    col_indices: list[int] = []
    for bid in box_ids:
        box = fmnn.boxes[bid]
        col = id_to_idx[bid]
        for u in box.members:
            row_indices.append(u)
            col_indices.append(col)

    if not row_indices:
        return {}

    data_vals = np.ones(len(row_indices), dtype=np.float32)
    B = sp.coo_matrix(
        (data_vals, (row_indices, col_indices)),
        shape=(num_nodes, num_boxes),
        dtype=np.float32,
    ).tocsr()

    C = B.T @ adj_csr @ B
    C_coo = C.tocoo()

    box_adj: dict[tuple[int, int], int] = {}
    for r, c, val in zip(C_coo.row, C_coo.col, C_coo.data):
        if r < c and val > 0:
            bid_i = box_ids[r]
            bid_j = box_ids[c]
            box_adj[(bid_i, bid_j)] = int(round(val))

    return box_adj


@dataclass
class BoxStats:
    """Cached structural and label statistics for a single hyperbox."""

    internal_edges: int
    boundary_edges: int
    volume: int
    conductance: float
    purity: float
    neighbor_agreement: float
    labeled_neighbors_count: int


class BoxGraphCache:
    """Cache holding structural graph statistics for all hyperboxes in an FMNN."""

    def __init__(
        self,
        adj_csr: sp.csr_matrix,
        degree: np.ndarray,
        y: np.ndarray | Any,
        train_mask: np.ndarray | Any,
    ) -> None:
        self.adj_csr = adj_csr
        self.degree = degree
        self.y = np.asarray(y)
        self.train_mask = np.asarray(train_mask)
        self.cache: dict[int, BoxStats] = {}

    def compute_box_stats(self, box: HyperBox) -> BoxStats:
        """Compute all graph statistics for a single hyperbox."""
        S = box.members
        intern = internal_edges(S, self.adj_csr)
        vol = volume(S, self.degree)
        bound = boundary_edges(S, self.adj_csr, degree=self.degree)
        cond = conductance(S, self.adj_csr, degree=self.degree)
        pur = class_purity(S, box.cls, self.y, self.train_mask)
        agr, count = neighbor_label_agreement(S, box.cls, self.adj_csr, self.y, self.train_mask)

        return BoxStats(
            internal_edges=intern,
            boundary_edges=bound,
            volume=vol,
            conductance=cond,
            purity=pur,
            neighbor_agreement=agr,
            labeled_neighbors_count=count,
        )

    def build_all(self, fmnn: FMNN) -> None:
        """Compute and store graph statistics for all hyperboxes."""
        self.cache.clear()
        for bid, box in fmnn.boxes.items():
            self.cache[bid] = self.compute_box_stats(box)

    def refresh(self, box_id: int, fmnn: FMNN) -> BoxStats:
        """Recompute and cache graph statistics for a modified hyperbox."""
        if box_id not in fmnn.boxes:
            raise KeyError(f"Box {box_id} does not exist in FMNN model")
        stats = self.compute_box_stats(fmnn.boxes[box_id])
        self.cache[box_id] = stats
        return stats

    def drop(self, box_id: int) -> None:
        """Remove a hyperbox from the statistics cache."""
        self.cache.pop(box_id, None)

    def get(self, box_id: int) -> BoxStats | None:
        """Retrieve cached statistics for a hyperbox."""
        return self.cache.get(box_id)
