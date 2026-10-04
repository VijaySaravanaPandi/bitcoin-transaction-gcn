"""
types.py — Core graph data types for Vanilla GCN.

All tensors follow PyTorch conventions (float32 for features/matrices,
long for labels and masks).

Notation
--------
N : int   — number of nodes
F : int   — number of input node features
C : int   — number of classes
E : int   — number of directed edges
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GraphData:
    """Container for a graph and its associated node-level data.

    All tensor attributes follow consistent dtype conventions:
    - Adjacency / normalized adjacency : ``torch.float32``
    - Feature matrix                   : ``torch.float32``
    - Labels                           : ``torch.long``
    - Masks                            : ``torch.bool``
    - Edge index                       : ``torch.long``

    Parameters
    ----------
    adjacency : torch.Tensor
        Raw (unnormalized) adjacency matrix A of shape ``(N, N)``.
        Should be a binary symmetric matrix (0/1 entries).
    features : torch.Tensor
        Node feature matrix X of shape ``(N, F)``.
        Each row is the feature vector for one node.
    labels : torch.Tensor
        Integer class labels y of shape ``(N,)``.
        Values in the range ``[0, C-1]``.
    node_ids : list[int]
        Ordered list of node identifiers, length N.
        Provides a human-readable mapping from row index → node name.
    edge_index : torch.Tensor
        COO edge list of shape ``(2, E)``.
        Row 0 = source nodes, row 1 = destination nodes.
    train_mask : torch.Tensor
        Boolean mask of shape ``(N,)``; ``True`` for training nodes.
    val_mask : torch.Tensor
        Boolean mask of shape ``(N,)``; ``True`` for validation nodes.
    test_mask : torch.Tensor
        Boolean mask of shape ``(N,)``; ``True`` for test nodes.
    num_classes : int
        Number of distinct classes C.
    name : str
        Human-readable dataset name (e.g. ``"synthetic"``).

    Notes
    -----
    The normalized adjacency ``A_tilde`` is NOT stored here — it is computed
    lazily inside :func:`vanilla_gcn.data.preprocessing.prepare_graph` and
    passed explicitly to the model forward pass.  This keeps the data
    container free of preprocessing concerns.
    """

    # ------------------------------------------------------------------
    # Core graph tensors
    # ------------------------------------------------------------------
    adjacency: torch.Tensor  # (N, N)  float32
    features: torch.Tensor   # (N, F)  float32
    labels: torch.Tensor     # (N,)    long

    # ------------------------------------------------------------------
    # Node identifiers
    # ------------------------------------------------------------------
    node_ids: list[int] = field(default_factory=list)

    # Optional temporal value for each node, such as Elliptic++ time step.
    node_times: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Structural helpers
    # ------------------------------------------------------------------
    edge_index: torch.Tensor = field(
        default_factory=lambda: torch.zeros((2, 0), dtype=torch.long)
    )

    # ------------------------------------------------------------------
    # Train / val / test masks
    # ------------------------------------------------------------------
    train_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )
    val_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )
    test_mask: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool)
    )
    labelled_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    num_classes: int = 0
    name: str = "unnamed"

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Number of nodes N."""
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        """Number of input features F per node."""
        return int(self.features.shape[1])

    @property
    def num_edges(self) -> int:
        """Number of directed edges E (each undirected edge counted twice)."""
        return int(self.edge_index.shape[1])

    def summary(self) -> str:
        """Return a concise string summary of the graph data."""
        lines = [
            f"GraphData(name='{self.name}')",
            f"  Nodes (N)        : {self.num_nodes}",
            f"  Features (F)     : {self.num_features}",
            f"  Classes (C)      : {self.num_classes}",
            f"  Directed edges(E): {self.num_edges}",
            f"  Train nodes      : {int(self.train_mask.sum())}",
            f"  Val nodes        : {int(self.val_mask.sum())}",
            f"  Test nodes       : {int(self.test_mask.sum())}",
            f"  adjacency shape  : {tuple(self.adjacency.shape)}",
            f"  features shape   : {tuple(self.features.shape)}",
            f"  labels shape     : {tuple(self.labels.shape)}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
