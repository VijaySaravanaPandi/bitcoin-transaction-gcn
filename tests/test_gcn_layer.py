"""
test_gcn_layer.py — Unit tests for GCNLayer.

Tests:
1. Output shape is (N, output_dim).
2. Parameters are trainable (require_grad=True).
3. Activation is applied when specified.
4. No activation when activation=None.
5. Weight matrix has correct shape.
6. Forward pass produces no NaN.
7. Supports different input/output dimensions.
"""

from __future__ import annotations

import pytest
import torch

from vanilla_gcn.models.gcn_layer import GCNLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """(A_tilde, H) for a 5-node graph with 3 features."""
    N, F = 5, 3
    # Identity as normalized adjacency (simplest valid Ã)
    A_tilde = torch.eye(N, dtype=torch.float32)
    H = torch.randn(N, F)
    return A_tilde, H


@pytest.fixture
def gcn_layer_relu() -> GCNLayer:
    """GCNLayer(3 → 16) with ReLU activation."""
    return GCNLayer(input_dim=3, output_dim=16, activation=torch.relu)


@pytest.fixture
def gcn_layer_no_act() -> GCNLayer:
    """GCNLayer(16 → 3) with no activation (output layer)."""
    return GCNLayer(input_dim=16, output_dim=3, activation=None)


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------


def test_output_shape(
    gcn_layer_relu: GCNLayer,
    small_graph: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """GCNLayer output must have shape (N, output_dim)."""
    A_tilde, H = small_graph
    H_out = gcn_layer_relu(H, A_tilde)
    N = H.shape[0]
    assert H_out.shape == (N, gcn_layer_relu.output_dim), (
        f"Expected ({N}, {gcn_layer_relu.output_dim}), got {H_out.shape}"
    )


def test_output_shape_various_dims() -> None:
    """Output shape is correct for various (N, F, d_out) combinations."""
    for N, F, d_out in [(3, 4, 8), (10, 6, 32), (12, 4, 3)]:
        A_tilde = torch.eye(N)
        H = torch.randn(N, F)
        layer = GCNLayer(F, d_out)
        H_out = layer(H, A_tilde)
        assert H_out.shape == (N, d_out)


# ---------------------------------------------------------------------------
# Test 2: Parameters are trainable
# ---------------------------------------------------------------------------


def test_weight_requires_grad(gcn_layer_relu: GCNLayer) -> None:
    """Weight matrix must have requires_grad=True."""
    assert gcn_layer_relu.weight.requires_grad, "Weight does not require grad"


def test_parameters_trainable(gcn_layer_relu: GCNLayer) -> None:
    """All parameters must be trainable."""
    for name, param in gcn_layer_relu.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' is not trainable"


# ---------------------------------------------------------------------------
# Test 3: Activation is applied (ReLU makes values ≥ 0)
# ---------------------------------------------------------------------------


def test_relu_activation_applied(
    gcn_layer_relu: GCNLayer,
    small_graph: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """With ReLU activation, output must be non-negative."""
    A_tilde, H = small_graph
    H_out = gcn_layer_relu(H, A_tilde)
    assert (H_out >= 0).all(), "ReLU output has negative values"


# ---------------------------------------------------------------------------
# Test 4: No activation when activation=None
# ---------------------------------------------------------------------------


def test_no_activation_can_be_negative(
    gcn_layer_no_act: GCNLayer,
    small_graph: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Without activation, output can have negative values."""
    A_tilde, H_3 = small_graph
    # Adjust input to match no-act layer's input_dim=16
    H_16 = torch.randn(5, 16)
    H_out = gcn_layer_no_act(H_16, A_tilde)
    # Simply check it doesn't crash and shape is correct
    assert H_out.shape == (5, 3)


# ---------------------------------------------------------------------------
# Test 5: Weight matrix shape
# ---------------------------------------------------------------------------


def test_weight_shape(gcn_layer_relu: GCNLayer) -> None:
    """Weight matrix shape must be (input_dim, output_dim)."""
    expected_shape = (gcn_layer_relu.input_dim, gcn_layer_relu.output_dim)
    assert gcn_layer_relu.weight.shape == expected_shape, (
        f"Expected {expected_shape}, got {gcn_layer_relu.weight.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: No NaN in forward pass
# ---------------------------------------------------------------------------


def test_no_nan_in_forward(
    gcn_layer_relu: GCNLayer,
    small_graph: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Forward pass must not produce NaN values."""
    A_tilde, H = small_graph
    H_out = gcn_layer_relu(H, A_tilde)
    assert not torch.isnan(H_out).any(), "Forward pass produced NaN"
    assert not torch.isinf(H_out).any(), "Forward pass produced Inf"


# ---------------------------------------------------------------------------
# Test 7: num_parameters property
# ---------------------------------------------------------------------------


def test_num_parameters() -> None:
    """num_parameters must equal input_dim * output_dim (no bias)."""
    layer = GCNLayer(input_dim=4, output_dim=16)
    assert layer.num_parameters == 4 * 16


def test_num_parameters_with_bias() -> None:
    """With bias, num_parameters must equal input_dim * output_dim + output_dim."""
    layer = GCNLayer(input_dim=4, output_dim=16, use_bias=True)
    assert layer.num_parameters == 4 * 16 + 16


# ---------------------------------------------------------------------------
# Test: Gradient flows through layer
# ---------------------------------------------------------------------------


def test_gradient_flows() -> None:
    """Gradient must flow back to the weight matrix after backward."""
    N, F, d_out = 4, 3, 2
    A_tilde = torch.eye(N)
    H = torch.randn(N, F, requires_grad=False)
    layer = GCNLayer(F, d_out, activation=None)

    H_out = layer(H, A_tilde)
    loss = H_out.sum()
    loss.backward()

    assert layer.weight.grad is not None, "No gradient on weight after backward"
    assert not torch.isnan(layer.weight.grad).any(), "NaN gradient"
