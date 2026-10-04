"""
preprocessing.py — Graph preprocessing for Vanilla GCN.

Implements the exact preprocessing pipeline from:
    Kipf & Welling (2017) "Semi-Supervised Classification with Graph
    Convolutional Networks"

and described in:
    Hamilton, W.L. (2020) "Graph Representation Learning", Chapters 5 & 7.

Pipeline
--------
Given a raw adjacency matrix A and node features X:

1.  Add self-loops:
        Â = A + I_N

2.  Compute degree matrix of Â:
        D̂_ii = Σ_j Â_ij

3.  Compute D̂^(-1/2):
        [D̂^(-1/2)]_ii = 1 / sqrt(D̂_ii)

4.  Symmetrically normalize:
        Ã = D̂^(-1/2) Â D̂^(-1/2)

The resulting Ã is used in every GCN forward pass:
        H^(k) = σ( Ã H^(k-1) W^(k) )
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from vanilla_gcn.data.types import GraphData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Self-loops
# ---------------------------------------------------------------------------


def add_self_loops(A: torch.Tensor) -> torch.Tensor:
    """Add self-loops to an adjacency matrix.

    Computes:
        Â = A + I_N

    where ``I_N`` is the identity matrix of size N×N.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix of shape ``(N, N)``.  Need not be binary — any
        non-negative values are accepted.

    Returns
    -------
    torch.Tensor
        Adjacency matrix with self-loops Â, shape ``(N, N)``.

    Examples
    --------
    >>> import torch
    >>> A = torch.zeros(3, 3)
    >>> A[0, 1] = A[1, 0] = 1.0
    >>> A_hat = add_self_loops(A)
    >>> A_hat.diagonal()  # [1., 1., 1.]
    tensor([1., 1., 1.])
    """
    N = A.shape[0]
    if A.is_sparse:
        A = A.coalesce()
        loop_indices = torch.arange(N, device=A.device, dtype=torch.long)
        loop_index = torch.stack([loop_indices, loop_indices])
        loop_values = torch.ones(N, dtype=A.dtype, device=A.device)
        indices = torch.cat([A.indices(), loop_index], dim=1)
        values = torch.cat([A.values(), loop_values])
        A_hat = torch.sparse_coo_tensor(
            indices, values, size=A.shape, device=A.device
        ).coalesce()
    else:
        A_hat = A + torch.eye(N, dtype=A.dtype, device=A.device)
    logger.debug("add_self_loops: A shape %s → Â shape %s", tuple(A.shape), tuple(A_hat.shape))
    return A_hat


# ---------------------------------------------------------------------------
# Step 2: Degree matrix
# ---------------------------------------------------------------------------


def compute_degree_matrix(A: torch.Tensor) -> torch.Tensor:
    """Compute the degree matrix D̂ of adjacency matrix Â.

    The degree matrix is diagonal:
        D̂_ii = Σ_j Â_ij    (row-sum of Â)

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix (should already include self-loops if desired),
        shape ``(N, N)``.

    Returns
    -------
    torch.Tensor
        Diagonal degree matrix D̂ of shape ``(N, N)``.

    Examples
    --------
    >>> import torch
    >>> A_hat = torch.tensor([[1., 1.], [1., 1.]])
    >>> D = compute_degree_matrix(A_hat)
    >>> D.diagonal()  # [2., 2.]
    tensor([2., 2.])
    """
    degree: torch.Tensor = A.sum(dim=1)  # row-sums → (N,)
    D: torch.Tensor = torch.diag(degree)
    logger.debug(
        "compute_degree_matrix: degree range [%.4f, %.4f]",
        float(degree.min()),
        float(degree.max()),
    )
    return D


# ---------------------------------------------------------------------------
# Step 3: D̂^(-1/2)
# ---------------------------------------------------------------------------


def compute_inverse_sqrt_degree(D: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Compute D̂^(-1/2), the inverse square root of the degree matrix.

    For a diagonal matrix D̂ this is simply:
        [D̂^(-1/2)]_ii = 1 / sqrt(D̂_ii + ε)

    The small ``eps`` guards against division by zero for isolated nodes.

    Parameters
    ----------
    D : torch.Tensor
        Diagonal degree matrix of shape ``(N, N)``.
    eps : float
        Small constant for numerical stability.  Default ``1e-12``.

    Returns
    -------
    torch.Tensor
        Inverse square-root degree matrix D̂^(-1/2) of shape ``(N, N)``.

    Notes
    -----
    Isolated nodes (degree 0) receive 1/√ε rather than inf.  This is a
    numerically safe choice and consistent with how many GCN implementations
    handle singletons.
    """
    degree: torch.Tensor = D.diagonal()  # (N,)
    inv_sqrt: torch.Tensor = 1.0 / torch.sqrt(degree + eps)
    D_inv_sqrt: torch.Tensor = torch.diag(inv_sqrt)
    logger.debug(
        "compute_inverse_sqrt_degree: inv-sqrt range [%.4f, %.4f]",
        float(inv_sqrt.min()),
        float(inv_sqrt.max()),
    )
    return D_inv_sqrt


# ---------------------------------------------------------------------------
# Step 4: Symmetric normalization
# ---------------------------------------------------------------------------


def symmetric_normalize(A: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Symmetrically normalize an adjacency matrix.

    Computes:
        Ã = D̂^(-1/2) Â D̂^(-1/2)

    This is the normalized adjacency used in every GCN forward pass.

    Parameters
    ----------
    A : torch.Tensor
        Adjacency matrix (typically Â = A + I) of shape ``(N, N)``.
    eps : float
        Numerical stability constant passed to
        :func:`compute_inverse_sqrt_degree`.

    Returns
    -------
    torch.Tensor
        Symmetrically normalized adjacency matrix Ã of shape ``(N, N)``.

    Mathematical Derivation
    -----------------------
    Given Â and its degree matrix D̂:

        Ã_ij = Â_ij / sqrt(D̂_ii) / sqrt(D̂_jj)

    This is equivalent to the matrix product D̂^(-1/2) Â D̂^(-1/2).
    The symmetric normalization preserves the spectral properties of Â and
    ensures that node representations remain bounded across layers.

    Examples
    --------
    >>> import torch
    >>> A_hat = torch.tensor([[2., 1.], [1., 2.]], dtype=torch.float32)
    >>> A_tilde = symmetric_normalize(A_hat)
    >>> # Entries should be in (0, 1]
    >>> (A_tilde >= 0).all() and (A_tilde <= 1).all()
    tensor(True)
    """
    if A.is_sparse:
        A = A.coalesce()
        degree = torch.sparse.sum(A, dim=1).to_dense()
        inv_sqrt = 1.0 / torch.sqrt(degree + eps)
        values = (
            A.values()
            * inv_sqrt[A.indices()[0]]
            * inv_sqrt[A.indices()[1]]
        )
        A_tilde = torch.sparse_coo_tensor(
            A.indices(), values, size=A.shape, device=A.device
        ).coalesce()
    else:
        D = compute_degree_matrix(A)
        D_inv_sqrt = compute_inverse_sqrt_degree(D, eps=eps)
        # Ã = D̂^(-1/2) @ Â @ D̂^(-1/2)
        A_tilde = D_inv_sqrt @ A @ D_inv_sqrt
    values = A_tilde.values() if A_tilde.is_sparse else A_tilde
    logger.debug(
        "symmetric_normalize: Ã shape %s, value range [%.4f, %.4f]",
        tuple(A_tilde.shape),
        float(values.min()),
        float(values.max()),
    )
    return A_tilde


# ---------------------------------------------------------------------------
# Step 5: Full pipeline
# ---------------------------------------------------------------------------


def prepare_graph(
    A: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the full GCN preprocessing pipeline.

    Pipeline
    --------
    1. Add self-loops:           Â  = A + I
    2. Degree matrix:            D̂  = diag(Â @ 1)
    3. Inverse sqrt degree:      D̂^(-1/2)
    4. Symmetric normalization:  Ã  = D̂^(-1/2) Â D̂^(-1/2)

    Parameters
    ----------
    A : torch.Tensor
        Raw binary adjacency matrix of shape ``(N, N)``.
    X : torch.Tensor
        Node feature matrix of shape ``(N, F)``.
    y : torch.Tensor
        Node label tensor of shape ``(N,)``.
    eps : float
        Numerical stability constant.

    Returns
    -------
    A_hat : torch.Tensor
        Adjacency with self-loops ``(N, N)``.
    A_tilde : torch.Tensor
        Symmetrically normalized adjacency ``(N, N)``.
    X : torch.Tensor
        Node features (passed through unchanged) ``(N, F)``.
    y : torch.Tensor
        Node labels (passed through unchanged) ``(N,)``.

    Examples
    --------
    >>> import torch
    >>> A = torch.eye(4)
    >>> X = torch.randn(4, 3)
    >>> y = torch.zeros(4, dtype=torch.long)
    >>> A_hat, A_tilde, X_out, y_out = prepare_graph(A, X, y)
    >>> A_hat.shape, A_tilde.shape
    (torch.Size([4, 4]), torch.Size([4, 4]))
    """
    logger.info("prepare_graph: starting GCN preprocessing pipeline")
    logger.info("  Input A shape : %s", tuple(A.shape))
    logger.info("  Input X shape : %s", tuple(X.shape))
    logger.info("  Input y shape : %s", tuple(y.shape))

    # Step 1
    A_hat = add_self_loops(A)
    logger.info("  Â (after self-loops) shape : %s", tuple(A_hat.shape))

    # Steps 2–4
    A_tilde = symmetric_normalize(A_hat, eps=eps)
    logger.info("  Ã (normalized) shape : %s", tuple(A_tilde.shape))

    return A_hat, A_tilde, X, y


# ---------------------------------------------------------------------------
# Utility: numpy → torch helpers
# ---------------------------------------------------------------------------


def numpy_to_tensor_graph(
    A: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to the expected torch tensors for GCN.

    Parameters
    ----------
    A : np.ndarray  shape (N, N)  — binary adjacency matrix
    X : np.ndarray  shape (N, F)  — node features
    y : np.ndarray  shape (N,)    — integer labels

    Returns
    -------
    A_t : torch.Tensor  float32  (N, N)
    X_t : torch.Tensor  float32  (N, F)
    y_t : torch.Tensor  long     (N,)
    """
    A_t = torch.tensor(A, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    return A_t, X_t, y_t
