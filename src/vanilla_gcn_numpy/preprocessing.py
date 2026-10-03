"""
preprocessing.py — Graph preprocessing (NumPy-only version).

Identical mathematical pipeline to the PyTorch version:
    Â  = A + I          (self-loops)
    D̂  = diag(row-sums) (degree matrix of Â)
    Ã  = D̂^(-1/2) Â D̂^(-1/2)  (symmetric normalization)
"""

from __future__ import annotations
import numpy as np


def add_self_loops(A: np.ndarray) -> np.ndarray:
    """Return Â = A + I."""
    return A + np.eye(A.shape[0], dtype=A.dtype)


def compute_degree_matrix(A_hat: np.ndarray) -> np.ndarray:
    """Return diagonal degree matrix D̂ where D̂_ii = Σ_j Â_ij."""
    return np.diag(A_hat.sum(axis=1))


def compute_inverse_sqrt_degree(D: np.ndarray) -> np.ndarray:
    """Return D̂^(-1/2).  Zero-degree nodes get 0 (safe)."""
    diag = D.diagonal().copy()
    inv_sqrt = np.where(diag > 0, 1.0 / np.sqrt(diag), 0.0)
    return np.diag(inv_sqrt)


def symmetric_normalize(A_hat: np.ndarray) -> np.ndarray:
    """Return Ã = D̂^(-1/2) Â D̂^(-1/2)."""
    D = compute_degree_matrix(A_hat)
    D_inv_sqrt = compute_inverse_sqrt_degree(D)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def prepare_graph(
    A: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full preprocessing pipeline.

    Returns
    -------
    A_hat   : Â = A + I            (N, N) float64
    A_tilde : Ã = D̂^(-1/2) Â D̂^(-1/2)  (N, N) float64
    X       : feature matrix       (N, F) float64
    y       : label vector         (N,)   int64
    """
    A = A.astype(np.float64)
    X = X.astype(np.float64)
    y = y.astype(np.int64)

    A_hat = add_self_loops(A)
    A_tilde = symmetric_normalize(A_hat)
    return A_hat, A_tilde, X, y
