"""
test_preprocessing.py — Unit tests for data preprocessing functions.

Tests:
1. Adjacency has correct dimensions.
2. Self-loops are correctly added (diagonal = 1 after add_self_loops).
3. Degree matrix is correct (row sums of Â).
4. Inverse sqrt degree is numerically stable.
5. Normalized adjacency has correct dimensions.
6. Normalized adjacency values are in [0, 1].
7. Symmetric normalization is symmetric (Ã = Ã^T).
8. prepare_graph returns correct shapes.
9. Zero-degree node safety (no inf/nan).
"""

from __future__ import annotations

import pytest
import torch

from vanilla_gcn.data.preprocessing import (
    add_self_loops,
    compute_degree_matrix,
    compute_inverse_sqrt_degree,
    prepare_graph,
    symmetric_normalize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_adjacency() -> torch.Tensor:
    """3-node path graph: 0—1—2."""
    A = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    return A


@pytest.fixture
def identity_adjacency() -> torch.Tensor:
    """4-node graph with only self-loops (identity matrix)."""
    return torch.eye(4, dtype=torch.float32)


@pytest.fixture
def four_node_graph() -> torch.Tensor:
    """4-node fully connected graph (minus diagonal)."""
    A = torch.ones(4, 4) - torch.eye(4)
    return A.float()


# ---------------------------------------------------------------------------
# Test 1: Correct adjacency dimensions
# ---------------------------------------------------------------------------


def test_adjacency_shape(simple_adjacency: torch.Tensor) -> None:
    """Adjacency matrix must be square."""
    A = simple_adjacency
    assert A.shape[0] == A.shape[1], "Adjacency must be square"
    assert A.ndim == 2, "Adjacency must be 2-dimensional"


# ---------------------------------------------------------------------------
# Test 2: Self-loops correctly added
# ---------------------------------------------------------------------------


def test_add_self_loops_diagonal(simple_adjacency: torch.Tensor) -> None:
    """After add_self_loops, diagonal must be 1."""
    A_hat = add_self_loops(simple_adjacency)
    diagonal = A_hat.diagonal()
    assert torch.all(diagonal == 1.0), f"Expected all-ones diagonal, got {diagonal}"


def test_add_self_loops_preserves_edges(simple_adjacency: torch.Tensor) -> None:
    """Off-diagonal entries must remain unchanged after self-loop addition."""
    A = simple_adjacency
    A_hat = add_self_loops(A)
    # Check upper-triangle matches
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j:
                assert A_hat[i, j] == A[i, j], (
                    f"Edge ({i},{j}) changed: {A[i,j]} → {A_hat[i,j]}"
                )


def test_add_self_loops_shape(simple_adjacency: torch.Tensor) -> None:
    """Output shape must match input shape."""
    A_hat = add_self_loops(simple_adjacency)
    assert A_hat.shape == simple_adjacency.shape


# ---------------------------------------------------------------------------
# Test 3: Degree matrix is correct
# ---------------------------------------------------------------------------


def test_degree_matrix_values(simple_adjacency: torch.Tensor) -> None:
    """Degree matrix diagonal must equal row sums of input."""
    A_hat = add_self_loops(simple_adjacency)
    D = compute_degree_matrix(A_hat)

    expected_degrees = A_hat.sum(dim=1)
    actual_degrees = D.diagonal()

    assert torch.allclose(actual_degrees, expected_degrees), (
        f"Degree mismatch: expected {expected_degrees}, got {actual_degrees}"
    )


def test_degree_matrix_is_diagonal(simple_adjacency: torch.Tensor) -> None:
    """All off-diagonal entries of D must be zero."""
    A_hat = add_self_loops(simple_adjacency)
    D = compute_degree_matrix(A_hat)
    N = D.shape[0]
    off_diagonal = D - torch.diag(D.diagonal())
    assert torch.all(off_diagonal == 0.0), "Degree matrix is not diagonal"


def test_degree_matrix_shape(four_node_graph: torch.Tensor) -> None:
    """Degree matrix shape must be (N, N)."""
    D = compute_degree_matrix(four_node_graph)
    N = four_node_graph.shape[0]
    assert D.shape == (N, N)


# ---------------------------------------------------------------------------
# Test 4: Inverse sqrt degree is numerically stable
# ---------------------------------------------------------------------------


def test_inverse_sqrt_degree_no_nan(simple_adjacency: torch.Tensor) -> None:
    """D^(-1/2) must not contain NaN or Inf."""
    A_hat = add_self_loops(simple_adjacency)
    D = compute_degree_matrix(A_hat)
    D_inv_sqrt = compute_inverse_sqrt_degree(D)

    assert not torch.isnan(D_inv_sqrt).any(), "D^(-1/2) contains NaN"
    assert not torch.isinf(D_inv_sqrt).any(), "D^(-1/2) contains Inf"


def test_inverse_sqrt_degree_zero_node() -> None:
    """Isolated (zero-degree) node must not cause NaN or Inf."""
    # Node 2 has no connections
    A = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    A_hat = add_self_loops(A)
    D = compute_degree_matrix(A_hat)
    D_inv_sqrt = compute_inverse_sqrt_degree(D)

    assert not torch.isnan(D_inv_sqrt).any()
    assert not torch.isinf(D_inv_sqrt).any()


def test_inverse_sqrt_degree_values(simple_adjacency: torch.Tensor) -> None:
    """D^(-1/2) diagonal values must equal 1/sqrt(degree)."""
    A_hat = add_self_loops(simple_adjacency)
    D = compute_degree_matrix(A_hat)
    D_inv_sqrt = compute_inverse_sqrt_degree(D)

    degree = D.diagonal()
    expected = 1.0 / torch.sqrt(degree)
    actual = D_inv_sqrt.diagonal()
    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5 & 6: Normalized adjacency dimensions and value range
# ---------------------------------------------------------------------------


def test_symmetric_normalize_shape(simple_adjacency: torch.Tensor) -> None:
    """Ã must have the same shape as Â."""
    A_hat = add_self_loops(simple_adjacency)
    A_tilde = symmetric_normalize(A_hat)
    assert A_tilde.shape == A_hat.shape


def test_symmetric_normalize_values_in_range(four_node_graph: torch.Tensor) -> None:
    """Ã entries must be in [0, 1] for non-negative adjacency."""
    A_hat = add_self_loops(four_node_graph)
    A_tilde = symmetric_normalize(A_hat)
    assert (A_tilde >= 0).all(), "Ã has negative values"
    assert (A_tilde <= 1.0 + 1e-6).all(), f"Ã has values > 1: max={A_tilde.max()}"


# ---------------------------------------------------------------------------
# Test 7: Symmetric normalization is symmetric
# ---------------------------------------------------------------------------


def test_symmetric_normalize_is_symmetric(four_node_graph: torch.Tensor) -> None:
    """Ã must be symmetric: Ã_ij == Ã_ji."""
    A_hat = add_self_loops(four_node_graph)
    A_tilde = symmetric_normalize(A_hat)
    assert torch.allclose(A_tilde, A_tilde.T, atol=1e-6), "Ã is not symmetric"


# ---------------------------------------------------------------------------
# Test 8: prepare_graph returns correct shapes
# ---------------------------------------------------------------------------


def test_prepare_graph_shapes() -> None:
    """prepare_graph must return tensors of the correct shapes."""
    N, F = 5, 3
    A = torch.zeros(N, N)
    A[0, 1] = A[1, 0] = 1.0
    X = torch.randn(N, F)
    y = torch.zeros(N, dtype=torch.long)

    A_hat, A_tilde, X_out, y_out = prepare_graph(A, X, y)

    assert A_hat.shape == (N, N),   f"A_hat shape mismatch: {A_hat.shape}"
    assert A_tilde.shape == (N, N), f"A_tilde shape mismatch: {A_tilde.shape}"
    assert X_out.shape == (N, F),   f"X shape mismatch: {X_out.shape}"
    assert y_out.shape == (N,),     f"y shape mismatch: {y_out.shape}"


def test_prepare_graph_no_nan() -> None:
    """prepare_graph must produce no NaN or Inf in any output."""
    N, F = 8, 4
    A = torch.zeros(N, N)
    for i in range(N - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    X = torch.randn(N, F)
    y = torch.zeros(N, dtype=torch.long)

    A_hat, A_tilde, _, _ = prepare_graph(A, X, y)

    for name, tensor in [("A_hat", A_hat), ("A_tilde", A_tilde)]:
        assert not torch.isnan(tensor).any(), f"{name} contains NaN"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf"
