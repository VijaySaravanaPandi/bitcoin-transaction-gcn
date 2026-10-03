"""
gcn_layer.py — Single Graph Convolutional Network (GCN) layer.

Implements the core GCN message-passing equation from:
    Kipf & Welling (2017) "Semi-Supervised Classification with Graph
    Convolutional Networks" — https://arxiv.org/abs/1609.02907

and described in:
    Hamilton (2020) "Graph Representation Learning", Chapters 5 & 7.

Core equation
-------------
Given:
    Ã  — symmetrically normalized adjacency matrix (N × N)
    H  — input node representations              (N × d_in)
    W  — learnable weight matrix                 (d_in × d_out)
    σ  — activation function (e.g. ReLU)

One GCN layer computes:
    H_out = σ( Ã H W )

Step-by-step:
    support = Ã @ H          # (N × d_in)  — neighborhood aggregation
    output  = support @ W    # (N × d_out) — linear transformation
    H_out   = σ(output)      # (N × d_out) — non-linear activation

Note on bias
------------
The minimal GCN equation in Hamilton Ch. 5 does NOT include a bias term.
This implementation optionally supports a bias vector b, which is an
implementation extension. The bias is disabled by default to stay faithful
to the original formulation. When enabled, it shifts the output:
    H_out = σ( Ã H W + b )
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    """A single vanilla GCN layer implementing H_out = σ( Ã H W ).

    Parameters
    ----------
    input_dim : int
        Dimensionality of input node features d_in.
    output_dim : int
        Dimensionality of output node embeddings d_out.
    activation : callable or None
        Activation function σ applied element-wise after the linear
        transformation.  Pass ``None`` for no activation (output layer).
        Default: ``torch.relu``.
    use_bias : bool
        If ``True``, add a learnable bias term b.  Not part of the minimal
        GCN equation — disabled by default.
    debug : bool
        If ``True``, print shape/stat information at every forward pass.

    Attributes
    ----------
    weight : nn.Parameter
        Learnable weight matrix W of shape ``(input_dim, output_dim)``.
    bias : nn.Parameter or None
        Learnable bias vector b of shape ``(output_dim,)`` or ``None``.

    Examples
    --------
    >>> layer = GCNLayer(4, 16, activation=torch.relu)
    >>> A_tilde = torch.eye(12)
    >>> H = torch.randn(12, 4)
    >>> H_out = layer(H, A_tilde)
    >>> H_out.shape
    torch.Size([12, 16])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.relu,
        use_bias: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.debug = debug

        # ------------------------------------------------------------------
        # Learnable weight matrix W ∈ R^(d_in × d_out)
        # ------------------------------------------------------------------
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))

        # ------------------------------------------------------------------
        # Optional bias (implementation extension, not in minimal GCN eq.)
        # ------------------------------------------------------------------
        if use_bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(output_dim))
        else:
            # Use register_parameter exclusively — do NOT also assign self.bias = None,
            # because nn.Module.__setattr__ already registers it, and a second
            # register_parameter call would raise KeyError in PyTorch 2.x+.
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weight matrix using Glorot (Xavier) uniform initialisation.

        Xavier initialisation keeps the variance of activations approximately
        constant across layers, which is important for training deep GCNs.

        Reference:
            Glorot & Bengio (2010) "Understanding the difficulty of training
            deep feedforward neural networks."
        """
        # fan_in = input_dim,  fan_out = output_dim
        std = math.sqrt(2.0 / (self.input_dim + self.output_dim))
        bound = math.sqrt(3.0) * std  # uniform(-bound, bound)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------
    # Forward pass — the core GCN computation
    # ------------------------------------------------------------------

    def forward(self, H: torch.Tensor, A_tilde: torch.Tensor) -> torch.Tensor:
        """Compute one GCN layer: H_out = σ( Ã H W ).

        Parameters
        ----------
        H : torch.Tensor, shape (N, d_in)
            Input node embedding matrix.  At layer k=1 this is the raw
            feature matrix X.
        A_tilde : torch.Tensor, shape (N, N)
            Symmetrically normalized adjacency matrix with self-loops.
            Pre-computed by :func:`vanilla_gcn.data.preprocessing.prepare_graph`.

        Returns
        -------
        torch.Tensor, shape (N, d_out)
            Output node embedding matrix after aggregation, linear
            transformation, and (optional) activation.

        Notes
        -----
        The computation is deliberately split into two explicit steps:

            support = Ã @ H      ← neighborhood aggregation
            output  = support @ W ← linear transformation

        This makes the matrix algebra transparent and matches the notation
        in Hamilton (2020) exactly.
        """
        if self.debug:
            self._debug_print(H, A_tilde)

        # ---------------------------------------------------------------
        # Step 1: Neighbourhood aggregation
        #   support = Ã @ H
        #   Shape: (N, N) @ (N, d_in) → (N, d_in)
        # ---------------------------------------------------------------
        support: torch.Tensor = A_tilde @ H

        # ---------------------------------------------------------------
        # Step 2: Linear transformation
        #   output = support @ W
        #   Shape: (N, d_in) @ (d_in, d_out) → (N, d_out)
        # ---------------------------------------------------------------
        output: torch.Tensor = support @ self.weight

        # ---------------------------------------------------------------
        # Step 3: Optional bias
        # ---------------------------------------------------------------
        if self.bias is not None:
            output = output + self.bias

        # ---------------------------------------------------------------
        # Step 4: Activation σ
        # ---------------------------------------------------------------
        if self.activation is not None:
            output = self.activation(output)

        return output

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def _debug_print(self, H: torch.Tensor, A_tilde: torch.Tensor) -> None:
        """Print shape and statistics for the current forward pass."""
        logger.debug("=" * 60)
        logger.debug("GCNLayer forward pass (input_dim=%d, output_dim=%d)", self.input_dim, self.output_dim)
        logger.debug("  Ã  shape  : %s", tuple(A_tilde.shape))
        logger.debug("  H  shape  : %s", tuple(H.shape))
        logger.debug("  W  shape  : %s", tuple(self.weight.shape))
        logger.debug(
            "  H  stats  : min=%.4f max=%.4f mean=%.4f",
            float(H.min()), float(H.max()), float(H.mean()),
        )
        logger.debug(
            "  W  stats  : min=%.4f max=%.4f mean=%.4f",
            float(self.weight.min()), float(self.weight.max()), float(self.weight.mean()),
        )
        support_shape = (A_tilde.shape[0], H.shape[1])
        output_shape = (A_tilde.shape[0], self.output_dim)
        logger.debug("  support shape (Ã@H)  : %s", support_shape)
        logger.debug("  output  shape (Ã@H@W): %s", output_shape)
        logger.debug("=" * 60)

    # ------------------------------------------------------------------
    # Extra utilities
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return extra info for nn.Module string representation."""
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"activation={self.activation.__name__ if self.activation else 'None'}, "
            f"bias={self.bias is not None}"
        )

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters in this layer."""
        total = self.weight.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total
