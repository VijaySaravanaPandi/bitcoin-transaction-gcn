"""
synthetic.py — Synthetic graph generator (NumPy-only version).

Identical dataset to vanilla_gcn.data.synthetic — 12-node, 3-community
graph with class-specific Gaussian features.  Returns raw NumPy arrays
instead of GraphData / torch.Tensor so the NumPy package is fully
self-sufficient (zero PyTorch dependency).
"""

from __future__ import annotations

import numpy as np


def create_synthetic_graph_numpy(
    seed: int = 42,
    n_per_class: int = 4,
    n_classes: int = 3,
    n_features: int = 4,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict[str, np.ndarray]:
    """Create a small synthetic community graph as NumPy arrays.

    Graph structure
    ---------------
    - N = n_per_class × n_classes nodes arranged in ``n_classes`` dense
      communities.
    - Within each community every pair of nodes is connected (complete
      sub-graph).
    - A small number of inter-community bridge edges are added.

    Node features
    -------------
    Each class has a different Gaussian mean.
    Features are drawn from  𝒩(μ_c, 0.3²).

    Parameters
    ----------
    seed        : int    — reproducibility seed
    n_per_class : int    — nodes per community
    n_classes   : int    — number of communities / classes
    n_features  : int    — feature dimension
    train_ratio : float  — fraction of nodes for training
    val_ratio   : float  — fraction of nodes for validation

    Returns
    -------
    dict with keys:
        ``A``          (N, N) float64 — binary adjacency matrix
        ``X``          (N, F) float64 — node feature matrix
        ``y``          (N,)   int64   — class labels  [0 … C-1]
        ``train_mask`` (N,)   bool
        ``val_mask``   (N,)   bool
        ``test_mask``  (N,)   bool
    """
    rng = np.random.default_rng(seed)
    N = n_per_class * n_classes

    # ── Adjacency ────────────────────────────────────────────────────────────
    A = np.zeros((N, N), dtype=np.float64)

    # Dense within-community edges
    for c in range(n_classes):
        start = c * n_per_class
        for i in range(start, start + n_per_class):
            for j in range(start, start + n_per_class):
                if i != j:
                    A[i, j] = 1.0

    # Sparse inter-community bridges (same as PyTorch version)
    bridges = [(0, 4), (4, 8), (1, 9), (3, 7)]
    for u, v in bridges:
        if u < N and v < N:
            A[u, v] = A[v, u] = 1.0

    # ── Features ─────────────────────────────────────────────────────────────
    means = rng.uniform(-2.0, 2.0, size=(n_classes, n_features))
    blocks = [
        rng.normal(means[c], 0.3, size=(n_per_class, n_features))
        for c in range(n_classes)
    ]
    X = np.vstack(blocks).astype(np.float64)

    # ── Labels ────────────────────────────────────────────────────────────────
    y = np.repeat(np.arange(n_classes, dtype=np.int64), n_per_class)

    # ── Masks ─────────────────────────────────────────────────────────────────
    idx = rng.permutation(N)
    n_train = max(1, int(N * train_ratio))
    n_val   = max(1, int(N * val_ratio))
    n_test  = N - n_train - n_val

    train_mask = np.zeros(N, dtype=bool)
    val_mask   = np.zeros(N, dtype=bool)
    test_mask  = np.zeros(N, dtype=bool)

    train_mask[idx[:n_train]]                     = True
    val_mask  [idx[n_train : n_train + n_val]]    = True
    test_mask [idx[n_train + n_val :]]            = True

    return {
        "A":          A,
        "X":          X,
        "y":          y,
        "train_mask": train_mask,
        "val_mask":   val_mask,
        "test_mask":  test_mask,
        "num_nodes":    N,
        "num_features": n_features,
        "num_classes":  n_classes,
    }
