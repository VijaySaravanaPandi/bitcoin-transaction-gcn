"""
test_model.py — Unit tests for VanillaGCN model.

Tests:
1. Model output shape is (N, num_classes).
2. return_intermediate=True returns all layer keys.
3. Parameters are trainable.
4. Single-layer model works.
5. Deep model (5 layers) works.
6. Architecture summary is non-empty.
7. count_parameters is positive.
8. get_embeddings returns correct keys.
"""

from __future__ import annotations

import pytest
import torch

from vanilla_gcn.data.preprocessing import prepare_graph
from vanilla_gcn.data.synthetic import create_synthetic_graph
from vanilla_gcn.models.vanilla_gcn import VanillaGCN, GCNIntermediateOutputs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Return preprocessed synthetic graph tensors."""
    data = create_synthetic_graph(seed=42)
    A_hat, A_tilde, X, y = prepare_graph(data.adjacency, data.features, data.labels)
    return data, A_tilde, X, y


@pytest.fixture
def base_model(synthetic_data) -> VanillaGCN:
    """Standard 2-layer VanillaGCN for the synthetic dataset."""
    data, A_tilde, X, y = synthetic_data
    return VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_classes=data.num_classes,
        num_layers=2,
    )


# ---------------------------------------------------------------------------
# Test 1: Output shape (N, num_classes)
# ---------------------------------------------------------------------------


def test_output_shape(base_model: VanillaGCN, synthetic_data) -> None:
    """Model output must be (N, num_classes)."""
    data, A_tilde, X, y = synthetic_data
    logits = base_model(X, A_tilde)
    N = data.num_nodes
    C = data.num_classes
    assert logits.shape == (N, C), f"Expected ({N}, {C}), got {logits.shape}"


# ---------------------------------------------------------------------------
# Test 2: return_intermediate returns all layer keys
# ---------------------------------------------------------------------------


def test_return_intermediate_keys(base_model: VanillaGCN, synthetic_data) -> None:
    """With return_intermediate=True, keys must include 'input', layers, 'logits'."""
    data, A_tilde, X, y = synthetic_data
    out = base_model(X, A_tilde, return_intermediate=True)

    assert isinstance(out, GCNIntermediateOutputs)
    keys = out.keys()
    assert "input" in keys, "Missing 'input' key"
    assert "logits" in keys or any("layer" in k for k in keys), "Missing layer keys"


def test_intermediate_shapes(base_model: VanillaGCN, synthetic_data) -> None:
    """All intermediate tensors must have N rows."""
    data, A_tilde, X, y = synthetic_data
    out = base_model(X, A_tilde, return_intermediate=True)
    N = data.num_nodes

    for key, tensor in out.items():
        assert tensor.shape[0] == N, f"Key '{key}' has wrong row count: {tensor.shape}"


# ---------------------------------------------------------------------------
# Test 3: Parameters are trainable
# ---------------------------------------------------------------------------


def test_parameters_trainable(base_model: VanillaGCN) -> None:
    """All model parameters must require gradients."""
    for name, param in base_model.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' is frozen"


# ---------------------------------------------------------------------------
# Test 4: Single-layer model
# ---------------------------------------------------------------------------


def test_single_layer_model(synthetic_data) -> None:
    """A 1-layer model must output shape (N, num_classes) without error."""
    data, A_tilde, X, y = synthetic_data
    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_classes=data.num_classes,
        num_layers=1,
    )
    logits = model(X, A_tilde)
    assert logits.shape == (data.num_nodes, data.num_classes)


# ---------------------------------------------------------------------------
# Test 5: Deep model (5 layers)
# ---------------------------------------------------------------------------


def test_deep_model(synthetic_data) -> None:
    """A 5-layer model must output correct shape without error."""
    data, A_tilde, X, y = synthetic_data
    model = VanillaGCN(
        input_dim=data.num_features,
        hidden_dim=16,
        num_classes=data.num_classes,
        num_layers=5,
    )
    logits = model(X, A_tilde)
    assert logits.shape == (data.num_nodes, data.num_classes)


# ---------------------------------------------------------------------------
# Test 6: Architecture summary
# ---------------------------------------------------------------------------


def test_architecture_summary(base_model: VanillaGCN) -> None:
    """architecture_summary must return a non-empty string."""
    summary = base_model.architecture_summary()
    assert isinstance(summary, str) and len(summary) > 0


# ---------------------------------------------------------------------------
# Test 7: count_parameters is positive
# ---------------------------------------------------------------------------


def test_count_parameters(base_model: VanillaGCN) -> None:
    """Total parameter count must be positive."""
    count = base_model.count_parameters()
    assert count > 0, f"Parameter count is {count}"


def test_parameter_count_formula() -> None:
    """Parameter count for a 2-layer GCN:
    Layer 1: F × H  parameters
    Layer 2: H × C  parameters
    """
    F, H, C = 4, 16, 3
    model = VanillaGCN(input_dim=F, hidden_dim=H, num_classes=C, num_layers=2)
    expected = F * H + H * C
    assert model.count_parameters() == expected, (
        f"Expected {expected}, got {model.count_parameters()}"
    )


# ---------------------------------------------------------------------------
# Test 8: get_embeddings
# ---------------------------------------------------------------------------


def test_get_embeddings(base_model: VanillaGCN, synthetic_data) -> None:
    """get_embeddings must return a dict with 'input' and 'logits'."""
    data, A_tilde, X, y = synthetic_data
    embs = base_model.get_embeddings(X, A_tilde)
    assert isinstance(embs, dict)
    assert "input" in embs
    # At least one layer embedding
    assert len(embs) >= 2


# ---------------------------------------------------------------------------
# Test: num_layers < 1 raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_num_layers() -> None:
    """num_layers < 1 must raise ValueError."""
    with pytest.raises(ValueError, match="num_layers"):
        VanillaGCN(input_dim=4, hidden_dim=16, num_classes=3, num_layers=0)
