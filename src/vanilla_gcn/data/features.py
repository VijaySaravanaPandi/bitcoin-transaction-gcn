"""Feature engineering utilities for graph datasets."""

from __future__ import annotations

import numpy as np


def add_transaction_graph_features(
    features: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """Append finite log-scaled in-degree, out-degree, and total degree."""
    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float32)
    total_degree = in_degree + out_degree
    graph_features = np.log1p(
        np.column_stack([in_degree, out_degree, total_degree])
    ).astype(np.float32)
    return np.concatenate([features, graph_features], axis=1).astype(np.float32)
