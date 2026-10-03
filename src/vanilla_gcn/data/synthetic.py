"""
synthetic.py — Deterministic synthetic graph for Vanilla GCN experiments.

Creates a small (12-node, 3-class) structured graph whose topology makes
the GCN demonstration visually meaningful:

    * Three communities of 4 nodes each (classes 0, 1, 2).
    * Dense intra-community edges, sparse inter-community bridges.
    * Each node carries a 4-dimensional feature vector correlated with
      its class membership.

Setting seed=42 guarantees identical output across all runs.

Usage
-----
    from vanilla_gcn.data.synthetic import create_synthetic_graph
    data = create_synthetic_graph(seed=42)
    print(data)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from vanilla_gcn.data.types import GraphData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph topology
# ---------------------------------------------------------------------------
# 12 nodes, 3 classes (communities) of 4 nodes each.
#
#  Class 0: nodes 0, 1, 2, 3   — densely connected
#  Class 1: nodes 4, 5, 6, 7   — densely connected
#  Class 2: nodes 8, 9, 10, 11 — densely connected
#
# Bridge edges cross communities to make the graph connected:
#   0–4  (class-0 ↔ class-1)
#   3–7  (class-0 ↔ class-1)
#   4–8  (class-1 ↔ class-2)
#   7–11 (class-1 ↔ class-2)

_INTRA_EDGES: list[tuple[int, int]] = [
    # Community 0
    (0, 1), (1, 2), (2, 3), (0, 3), (1, 3),
    # Community 1
    (4, 5), (5, 6), (6, 7), (4, 7), (5, 7),
    # Community 2
    (8, 9), (9, 10), (10, 11), (8, 11), (9, 11),
]

_INTER_EDGES: list[tuple[int, int]] = [
    (0, 4), (3, 7),    # class-0 ↔ class-1
    (4, 8), (7, 11),   # class-1 ↔ class-2
]

_ALL_EDGES: list[tuple[int, int]] = _INTRA_EDGES + _INTER_EDGES

_LABELS: list[int] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]  # 12 nodes
_NUM_NODES: int = 12
_NUM_CLASSES: int = 3
_FEATURE_DIM: int = 4


def _build_adjacency() -> np.ndarray:
    """Construct the symmetric binary adjacency matrix."""
    A = np.zeros((_NUM_NODES, _NUM_NODES), dtype=np.float32)
    for u, v in _ALL_EDGES:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _build_features(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Construct node features correlated with class membership.

    Each node's feature vector is drawn from a class-specific Gaussian:
        class 0 → mean ≈ [1, 0, 0, 0]
        class 1 → mean ≈ [0, 1, 0, 0]
        class 2 → mean ≈ [0, 0, 1, 0]

    Noise is added so the problem is non-trivial but still solvable.
    """
    class_means = np.array(
        [
            [2.0,  0.0,  0.0,  0.5],   # class 0
            [0.0,  2.0,  0.0, -0.5],   # class 1
            [0.0,  0.0,  2.0,  0.0],   # class 2
        ],
        dtype=np.float32,
    )
    X = np.zeros((_NUM_NODES, _FEATURE_DIM), dtype=np.float32)
    for node_idx, label in enumerate(labels):
        X[node_idx] = class_means[label] + rng.normal(0.0, 0.3, size=_FEATURE_DIM)
    return X.astype(np.float32)


def _build_edge_index(A: np.ndarray) -> torch.Tensor:
    """Build COO edge index from adjacency matrix."""
    src, dst = np.nonzero(A)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    return edge_index


def create_synthetic_graph(
    seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> GraphData:
    """Create a deterministic synthetic graph for GCN experiments.

    Parameters
    ----------
    seed : int
        Random seed for reproducible feature generation.  Default ``42``.
    train_ratio : float
        Fraction of nodes in the training set.
    val_ratio : float
        Fraction of nodes in the validation set.
    test_ratio : float
        Fraction of nodes in the test set.

    Returns
    -------
    GraphData
        Fully populated graph data object ready for GCN training.

    Graph Properties
    ----------------
    - N = 12 nodes
    - F = 4  node features
    - C = 3  classes (communities)
    - Symmetric edges (undirected graph)
    - Class-correlated node features (Gaussian noise around class means)

    Notes
    -----
    The three-community structure means a well-trained GCN with 2+ layers
    should achieve near-perfect node classification on this graph.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    rng = np.random.default_rng(seed)

    labels_np = np.array(_LABELS, dtype=np.int64)
    A_np = _build_adjacency()
    X_np = _build_features(labels_np, rng)

    # Convert to torch
    A = torch.tensor(A_np, dtype=torch.float32)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(labels_np, dtype=torch.long)
    edge_index = _build_edge_index(A_np)

    # ------------------------------------------------------------------
    # Stratified split (equal samples per class)
    # ------------------------------------------------------------------
    n = _NUM_NODES
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Stratify: pick proportionally from each class
    all_indices = np.arange(n)
    rng2 = np.random.default_rng(seed)
    perm = rng2.permutation(all_indices)

    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = GraphData(
        adjacency=A,
        features=X,
        labels=y,
        node_ids=list(range(n)),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=_NUM_CLASSES,
        name="synthetic_community_graph",
    )

    logger.info("Created synthetic graph: %s", data.summary())
    return data
