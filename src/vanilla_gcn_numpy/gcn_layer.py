"""
gcn_layer.py — Single GCN layer with manual forward + backward (NumPy).

Architecture — exactly the same as the PyTorch version:

    Forward:
        S      = Ã @ H          (neighbour aggregation)
        Z      = S @ W          (linear projection)
        H_out  = σ(Z)           (activation)

    Backward (via chain rule):
        dL/dZ  = dL/dH_out ⊙ σ'(Z)
        dL/dW  = Sᵀ @ dL/dZ
        dL/dH  = Ãᵀ @ (dL/dZ @ Wᵀ)   [Ã is symmetric so Ãᵀ = Ã]

Weight initialisation: Glorot uniform  (same as PyTorch default for nn.Linear).
"""

from __future__ import annotations
import numpy as np


def _glorot_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class GCNLayerNumPy:
    """One GCN layer: H_out = σ(Ã @ H @ W).

    Parameters
    ----------
    input_dim  : int   — number of input features
    output_dim : int   — number of output features
    activation : str or None — 'relu' or None
    seed       : int   — for reproducible weight init
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str | None = "relu",
        seed: int = 42,
    ) -> None:
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.activation = activation

        rng = np.random.default_rng(seed)
        self.W: np.ndarray = _glorot_uniform(input_dim, output_dim, rng)
        self.dW: np.ndarray | None = None  # gradient, filled by backward()

        # Internal cache for backward pass
        self._cache: dict = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, H: np.ndarray, A_tilde: np.ndarray) -> np.ndarray:
        """Compute H_out = σ(Ã @ H @ W) and cache intermediates.

        Parameters
        ----------
        H       : (N, input_dim)  node embedding matrix
        A_tilde : (N, N)          symmetrically normalised adjacency

        Returns
        -------
        H_out   : (N, output_dim)
        """
        S = A_tilde @ H         # (N, input_dim)  — aggregation
        Z = S @ self.W          # (N, output_dim) — linear

        if self.activation == "relu":
            H_out = np.maximum(0.0, Z)
        else:
            H_out = Z           # no activation (output layer)

        # Cache for backward
        self._cache = {"H": H, "S": S, "Z": Z, "A_tilde": A_tilde}
        return H_out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dL_dH_out: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through this layer.

        Parameters
        ----------
        dL_dH_out : (N, output_dim)  upstream gradient

        Returns
        -------
        dL_dH     : (N, input_dim)   gradient w.r.t. this layer's input H
        """
        Z       = self._cache["Z"]
        S       = self._cache["S"]
        A_tilde = self._cache["A_tilde"]

        # σ'(Z): derivative of activation
        if self.activation == "relu":
            dL_dZ = dL_dH_out * (Z > 0)   # (N, output_dim)
        else:
            dL_dZ = dL_dH_out

        # Gradient w.r.t. weight matrix
        self.dW = S.T @ dL_dZ              # (input_dim, output_dim)

        # Gradient flowing back into H (through Ã — which is symmetric)
        dL_dS = dL_dZ @ self.W.T          # (N, input_dim)
        dL_dH = A_tilde.T @ dL_dS         # (N, input_dim)
        return dL_dH

    # ------------------------------------------------------------------
    # Parameter update (SGD with optional L2 regularisation)
    # ------------------------------------------------------------------

    def step(self, lr: float, weight_decay: float = 0.0) -> None:
        """Update W in-place using the cached gradient."""
        if self.dW is None:
            raise RuntimeError("Call backward() before step().")
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.dW = None

    @property
    def num_parameters(self) -> int:
        return int(self.W.size)

    def __repr__(self) -> str:
        return (
            f"GCNLayerNumPy(in={self.input_dim}, out={self.output_dim}, "
            f"act={self.activation}, params={self.num_parameters})"
        )
