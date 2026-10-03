"""
vanilla_gcn.py — K-layer Vanilla GCN (NumPy-only).

Same architecture as the PyTorch version:
    H^(0) = X
    H^(k) = σ( Ã H^(k-1) W^(k) )   k = 1 … K-1   [ReLU]
    H^(K) = Ã H^(K-1) W^(K)          [no activation → logits]
"""

from __future__ import annotations
import numpy as np
from vanilla_gcn_numpy.gcn_layer import GCNLayerNumPy


class VanillaGCNNumPy:
    """K-layer GCN implemented with pure NumPy.

    Parameters
    ----------
    input_dim   : int — input feature dimension F
    hidden_dim  : int — hidden layer dimension
    num_classes : int — number of output classes C
    num_layers  : int — total number of GCN layers (≥ 1)
    seed        : int — reproducibility seed
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        seed: int = 42,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes
        self.num_layers  = num_layers

        # Build layers — exactly matching the PyTorch VanillaGCN topology
        self.layers: list[GCNLayerNumPy] = []

        if num_layers == 1:
            # Single layer: input → classes (no activation)
            self.layers.append(GCNLayerNumPy(input_dim, num_classes, activation=None, seed=seed))
        else:
            # First hidden layer: input → hidden
            self.layers.append(GCNLayerNumPy(input_dim, hidden_dim, activation="relu", seed=seed))
            # Intermediate hidden layers
            for k in range(1, num_layers - 1):
                self.layers.append(GCNLayerNumPy(hidden_dim, hidden_dim, activation="relu", seed=seed + k))
            # Output layer: hidden → classes (no activation)
            self.layers.append(GCNLayerNumPy(hidden_dim, num_classes, activation=None, seed=seed + num_layers))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        X: np.ndarray,
        A_tilde: np.ndarray,
        return_intermediate: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Run the full forward pass.

        Parameters
        ----------
        X       : (N, F)  node features
        A_tilde : (N, N)  symmetrically normalised adjacency
        return_intermediate : bool — if True, return dict of all embeddings

        Returns
        -------
        logits  : (N, C)   or   dict mapping layer name → embedding
        """
        intermediates: dict[str, np.ndarray] = {}

        H = X.copy()
        if return_intermediate:
            intermediates["input"] = H.copy()

        for k, layer in enumerate(self.layers):
            H = layer.forward(H, A_tilde)
            if return_intermediate:
                name = f"layer_{k + 1}" if k < len(self.layers) - 1 else "logits"
                intermediates[name] = H.copy()

        if return_intermediate:
            if "logits" not in intermediates:
                intermediates["logits"] = H.copy()
            return intermediates

        return H  # logits: (N, C)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dL_dlogits: np.ndarray) -> None:
        """Backpropagate gradient through all layers (reverse order).

        Parameters
        ----------
        dL_dlogits : (N, C)  gradient of loss w.r.t. logits
        """
        grad = dL_dlogits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        """Update all layer weights (forward order, doesn't matter)."""
        for layer in self.layers:
            layer.step(lr, weight_decay)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self.layers)

    def get_embeddings(
        self, X: np.ndarray, A_tilde: np.ndarray
    ) -> dict[str, np.ndarray]:
        return self.forward(X, A_tilde, return_intermediate=True)

    def architecture_summary(self) -> str:
        lines = [
            "VanillaGCNNumPy (Pure NumPy — no PyTorch)",
            "=" * 45,
        ]
        for k, layer in enumerate(self.layers):
            lines.append(f"  Layer {k + 1}: {layer}")
        lines.append(f"  Total parameters: {self.count_parameters():,}")
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.architecture_summary())

    def __repr__(self) -> str:
        return (
            f"VanillaGCNNumPy(in={self.input_dim}, hidden={self.hidden_dim}, "
            f"out={self.num_classes}, layers={self.num_layers}, "
            f"params={self.count_parameters()})"
        )
