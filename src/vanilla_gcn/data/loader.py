"""
loader.py — Generic graph data loader for Vanilla GCN.

This module defines the dataset interface that allows the GCN model to
work with any real-world dataset without modifying the model architecture.

Future loaders (CSV, NPZ, NetworkX, citation networks, etc.) should be
added here as separate functions that all return a :class:`GraphData` object.

See also
--------
docs/dataset_interface.md — detailed instructions for adding new datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from vanilla_gcn.data.types import GraphData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic numpy loader — the primary interface for real-world datasets
# ---------------------------------------------------------------------------


def load_graph_from_numpy(
    A: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    name: str = "custom",
) -> GraphData:
    """Load a graph from NumPy arrays into a :class:`GraphData` object.

    This is the primary entry point for replacing the synthetic dataset
    with a real-world graph.  The GCN model, trainer, and evaluation code
    only depend on :class:`GraphData` — they do not know or care where the
    data comes from.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Binary symmetric adjacency matrix.  Off-diagonal entry A[i,j] = 1
        means there is an edge between node i and node j.
    X : np.ndarray, shape (N, F)
        Node feature matrix.  Row i is the feature vector for node i.
    y : np.ndarray, shape (N,)
        Integer class labels in the range [0, C-1].
    train_ratio : float
        Fraction of nodes assigned to the training set.
    val_ratio : float
        Fraction of nodes assigned to the validation set.
    test_ratio : float
        Fraction of nodes assigned to the test set.
    seed : int
        Random seed for the node split.
    name : str
        Human-readable dataset name stored in :class:`GraphData`.

    Returns
    -------
    GraphData
        Fully populated graph data container ready for GCN training.

    Examples
    --------
    Replace the synthetic graph with your own data:

    >>> import numpy as np
    >>> from vanilla_gcn.data.loader import load_graph_from_numpy
    >>> A = np.load("my_adjacency.npy")   # shape (N, N)
    >>> X = np.load("my_features.npy")    # shape (N, F)
    >>> y = np.load("my_labels.npy")      # shape (N,)
    >>> data = load_graph_from_numpy(A, X, y, name="my_dataset")

    Then pass ``data`` directly to the training pipeline — no other changes
    are required.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio + test_ratio must equal 1.0"
    )
    n = A.shape[0]
    num_classes = int(y.max()) + 1

    # Split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    # Edge index from adjacency
    src, dst = np.nonzero(A)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    data = GraphData(
        adjacency=torch.tensor(A, dtype=torch.float32),
        features=torch.tensor(X, dtype=torch.float32),
        labels=torch.tensor(y, dtype=torch.long),
        node_ids=list(range(n)),
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=num_classes,
        name=name,
    )

    logger.info("Loaded graph '%s': N=%d, F=%d, C=%d", name, n, X.shape[1], num_classes)
    return data


# ---------------------------------------------------------------------------
# Split utility
# ---------------------------------------------------------------------------


def split_nodes(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random train/val/test boolean masks.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes N.
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0).
    seed : int
        Random seed.

    Returns
    -------
    train_mask, val_mask, test_mask : torch.Tensor
        Boolean tensors of shape ``(N,)``.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[perm[:n_train]] = True
    val_mask[perm[n_train: n_train + n_val]] = True
    test_mask[perm[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------------
# NPZ loader (future-ready)
# ---------------------------------------------------------------------------


def load_graph_from_npz(
    path: str | Path,
    *,
    adjacency_key: str = "adjacency",
    features_key: str = "features",
    labels_key: str = "labels",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> GraphData:
    """Load a graph from an NPZ archive.

    Expected NPZ keys (configurable via keyword arguments):
    - ``adjacency`` : shape (N, N)
    - ``features``  : shape (N, F)
    - ``labels``    : shape (N,)

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.
    adjacency_key, features_key, labels_key : str
        Keys for the respective arrays inside the NPZ file.
    train_ratio, val_ratio, test_ratio : float
        Node split ratios.
    seed : int
        Random seed.

    Returns
    -------
    GraphData
        Populated graph data object.
    """
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    archive = np.load(npz_path, allow_pickle=False)
    A = archive[adjacency_key].astype(np.float32)
    X = archive[features_key].astype(np.float32)
    y = archive[labels_key].astype(np.int64)

    return load_graph_from_numpy(
        A, X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        name=npz_path.stem,
    )
