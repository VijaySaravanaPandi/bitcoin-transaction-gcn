"""
vanilla_gcn.py — Full Vanilla GCN model.

Stacks multiple GCNLayer modules to implement the K-layer GCN:

    H^(0) = X

    H^(k) = σ( Ã H^(k-1) W^(k) )   for k = 1, ..., K-1

    H^(K) = Ã H^(K-1) W^(K)         (no activation on last layer → logits)

For node classification:
    logits = H^(K)    ∈ R^(N × C)
    ŷ_u   = argmax_c logits[u, c]

Note
----
Softmax is NOT applied inside the model when CrossEntropyLoss is used,
because PyTorch's CrossEntropyLoss internally applies log-softmax.

Reference
---------
    Hamilton, W.L. (2020) "Graph Representation Learning"
    Chapters 5 and 7.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from vanilla_gcn.models.gcn_layer import GCNLayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate output container
# ---------------------------------------------------------------------------


class GCNIntermediateOutputs:
    """Container for intermediate layer representations.

    Attributes
    ----------
    embeddings : dict[str, torch.Tensor]
        Mapping from layer name to embedding tensor.
        Keys follow the pattern: ``'input'``, ``'layer_1'``, ..., ``'logits'``.

    Examples
    --------
    >>> out = model(X, A_tilde, return_intermediate=True)
    >>> out['input'].shape    # (N, F)
    >>> out['layer_1'].shape  # (N, hidden_dim)
    >>> out['logits'].shape   # (N, num_classes)
    """

    def __init__(self) -> None:
        self._data: dict[str, torch.Tensor] = {}

    def store(self, key: str, value: torch.Tensor) -> None:
        self._data[key] = value.detach()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def items(self) -> list[tuple[str, torch.Tensor]]:
        return list(self._data.items())

    def __repr__(self) -> str:
        parts = [f"  {k}: {tuple(v.shape)}" for k, v in self._data.items()]
        return "GCNIntermediateOutputs(\n" + "\n".join(parts) + "\n)"


# ---------------------------------------------------------------------------
# VanillaGCN
# ---------------------------------------------------------------------------


class VanillaGCN(nn.Module):
    """Vanilla Graph Convolutional Network for node classification.

    Architecture:

        Input (X)
            ↓
        GCN Layer 1  [input_dim  → hidden_dim]  + ReLU
            ↓
        GCN Layer 2  [hidden_dim → hidden_dim]  + ReLU   (if num_layers > 2)
            ↓
            ...
            ↓
        GCN Output   [hidden_dim → num_classes] (no activation → logits)

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw node features F.
    hidden_dim : int
        Dimensionality of hidden GCN layers.
    num_classes : int
        Number of output classes C.
    num_layers : int
        Total number of GCN layers K (including the output layer).
        Must be ≥ 1.  With ``num_layers=1`` the model is a linear graph
        filter; with ``num_layers=2`` it's the standard 2-layer GCN.
    dropout : float
        Dropout probability applied to hidden representations.
        Set to ``0.0`` to disable (default for vanilla baseline).
    debug : bool
        If ``True``, each GCNLayer prints debug information on each
        forward pass.

    Attributes
    ----------
    layers : nn.ModuleList
        List of :class:`GCNLayer` instances.

    Examples
    --------
    >>> model = VanillaGCN(input_dim=4, hidden_dim=16, num_classes=3)
    >>> logits = model(X, A_tilde)         # shape (N, 3)
    >>> out = model(X, A_tilde, return_intermediate=True)
    >>> out['layer_1'].shape               # (N, 16)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        debug: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be ≥ 1, got {num_layers}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.debug = debug

        # ------------------------------------------------------------------
        # Build layer stack
        # ------------------------------------------------------------------
        layer_list: list[GCNLayer] = []

        if num_layers == 1:
            # Single-layer model: input directly to output
            layer_list.append(
                GCNLayer(input_dim, num_classes, activation=None, debug=debug)
            )
        else:
            # First layer: input_dim → hidden_dim
            layer_list.append(
                GCNLayer(input_dim, hidden_dim, activation=torch.relu, debug=debug)
            )
            # Intermediate hidden layers (if num_layers > 2)
            for _ in range(num_layers - 2):
                layer_list.append(
                    GCNLayer(hidden_dim, hidden_dim, activation=torch.relu, debug=debug)
                )
            # Output layer: hidden_dim → num_classes (no activation)
            layer_list.append(
                GCNLayer(hidden_dim, num_classes, activation=None, debug=debug)
            )

        self.layers = nn.ModuleList(layer_list)

        logger.info(
            "VanillaGCN created: input_dim=%d, hidden_dim=%d, num_classes=%d, "
            "num_layers=%d, dropout=%.2f",
            input_dim, hidden_dim, num_classes, num_layers, dropout,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        A_tilde: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, GCNIntermediateOutputs]:
        """Run the full GCN forward pass.

        Computes:
            H^(0) = X
            H^(k) = σ( Ã H^(k-1) W^(k) )   for k = 1, ..., K

        Parameters
        ----------
        X : torch.Tensor, shape (N, F)
            Input node feature matrix H^(0) = X.
        A_tilde : torch.Tensor, shape (N, N)
            Symmetrically normalized adjacency with self-loops.
        return_intermediate : bool
            If ``True``, return a :class:`GCNIntermediateOutputs` object
            containing embeddings from every layer.
            If ``False`` (default), return only the final logit tensor.

        Returns
        -------
        torch.Tensor, shape (N, num_classes)
            Node classification logits (when ``return_intermediate=False``).
        GCNIntermediateOutputs
            Intermediate embeddings keyed by layer name
            (when ``return_intermediate=True``).
        """
        intermediate: Optional[GCNIntermediateOutputs] = None
        if return_intermediate:
            intermediate = GCNIntermediateOutputs()
            intermediate.store("input", X)

        H: torch.Tensor = X

        for k, layer in enumerate(self.layers):
            # Apply dropout to hidden representations (not input or output)
            if self.dropout > 0.0 and k > 0 and k < len(self.layers) - 1:
                H = F.dropout(H, p=self.dropout, training=self.training)

            H = layer(H, A_tilde)

            if intermediate is not None:
                key = f"layer_{k + 1}" if k < len(self.layers) - 1 else "logits"
                intermediate.store(key, H)

        # Rename last stored key to 'logits' for clarity
        if intermediate is not None:
            if f"layer_{len(self.layers)}" in intermediate:
                pass  # Already stored as logits if only 1 layer
            return intermediate

        return H  # shape (N, num_classes) — logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_embeddings(
        self, X: torch.Tensor, A_tilde: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return all intermediate embeddings as a plain dictionary.

        Parameters
        ----------
        X : torch.Tensor
            Input features.
        A_tilde : torch.Tensor
            Normalized adjacency.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: ``'input'``, ``'layer_1'``, ..., ``'logits'``.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(X, A_tilde, return_intermediate=True)
        if isinstance(out, GCNIntermediateOutputs):
            return dict(out.items())
        return {"logits": out}

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> str:
        """Return a human-readable architecture summary string.

        Example output::

            ┌─────────────────────────────────┐
            │        Vanilla GCN              │
            ├─────────────────────────────────┤
            │ Input                 (N × 4)   │
            │   ↓                             │
            │ GCN Layer 1  4 → 16  ReLU       │
            │   ↓                             │
            │ GCN Layer 2  16 → 3  None       │
            │   ↓                             │
            │ Logits               (N × 3)    │
            ├─────────────────────────────────┤
            │ Total parameters: 115           │
            └─────────────────────────────────┘
        """
        lines = [
            "┌─────────────────────────────────────────┐",
            "│              Vanilla GCN                │",
            "├─────────────────────────────────────────┤",
            f"│  Input features     (N × {self.input_dim})",
            "│     ↓",
        ]
        for k, layer in enumerate(self.layers):
            act_name = layer.activation.__name__ if layer.activation else "None"
            is_last = k == len(self.layers) - 1
            label = "Output Layer (Logits)" if is_last else f"GCN Layer {k + 1}"
            lines.append(
                f"│  {label:22s}  {layer.input_dim} → {layer.output_dim}  "
                f"activation={act_name}"
            )
            if not is_last:
                lines.append("│     ↓")
        lines.append("│     ↓")
        lines.append(f"│  Logits             (N × {self.num_classes})")
        lines.append("├─────────────────────────────────────────┤")
        lines.append(f"│  Total trainable parameters: {self.count_parameters()}")
        lines.append("└─────────────────────────────────────────┘")
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the architecture summary to stdout."""
        print(self.architecture_summary())

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_classes={self.num_classes}, num_layers={self.num_layers}"
        )
